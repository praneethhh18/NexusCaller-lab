"""
Voice-agent FastAPI server — outbound calling cockpit (LiveKit edition).

What it does
─────────────
NexusAgent's CRM hits POST /api/voice/prepare-dial → opens this server's
/precall page in a new tab. The operator picks a stack combo, clicks
Place call → /api/dial:

    1. Generates a fresh call_id and LiveKit room name.
    2. Dispatches a Vox agent worker job to that room (via LiveKit API),
       attaching the per-call metadata (contact, business, purpose, stack).
    3. Creates a SIP outbound participant — LiveKit dials the contact's
       phone number through the configured SIP trunk (Twilio Elastic SIP),
       which connects the PSTN audio into the same room as the agent.
    4. Returns a watch_url. The operator's browser redirects to /calls/<id>
       which renders the cockpit, connects to the same room as a passive
       viewer, and shows live transcript + summary via DataChannel.

Pipecat-era endpoints (POST /twilio/voice, WS /twilio/stream) are gone —
LiveKit owns all audio routing now, with built-in Krisp echo cancellation.

Endpoints
─────────
    GET  /health                   liveness check
    GET  /precall                  static HTML — combo picker
    GET  /api/catalog              combos + STT/LLM/TTS dropdown options
    POST /api/dial                 main entry — creates room + dispatches agent + SIP-dials phone
    GET  /calls/{call_id}          static HTML — live cockpit
    GET  /api/cockpit-token/{id}   issues a LiveKit JWT scoped to the room (subscribe-only)
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from livekit import api
from livekit.api import (
    AccessToken, CreateAgentDispatchRequest, CreateSIPParticipantRequest,
    LiveKitAPI, RoomConfiguration, VideoGrants,
)
from loguru import logger

from voice_agent.combos import (
    LLM_OPTIONS, PRESETS, STT_OPTIONS, TTS_OPTIONS, default_combo, find_combo,
)


load_dotenv()

app = FastAPI(title="NexusCaller Voice Agent")

ROOT = Path(__file__).parent
TRANSCRIPT_DIR = ROOT / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise HTTPException(
            500,
            f"{name} not set in .env — see voice_agent/README.md for setup.",
        )
    return val


# Supported telephony providers — each maps to a LiveKit outbound SIP trunk.
# Backward-compat: LIVEKIT_OUTBOUND_TRUNK_ID is the legacy Twilio var;
# we fall back to it when LIVEKIT_TRUNK_TWILIO isn't explicitly set.
_TELEPHONY = {
    "twilio": {
        "label": "Twilio",
        "desc":  "Elastic SIP Trunk · ~$0.013/min US · ~$0.20/min India",
        "env":   "LIVEKIT_TRUNK_TWILIO",
    },
    "telnyx": {
        "label": "Telnyx",
        "desc":  "~$0.002/min US · ~$0.05/min India · $10 free credit",
        "env":   "LIVEKIT_TRUNK_TELNYX",
    },
    "signalwire": {
        "label": "SignalWire",
        "desc":  "Twilio-compatible API · ~$0.02/min · $5 free credit",
        "env":   "LIVEKIT_TRUNK_SIGNALWIRE",
    },
    "exotel": {
        "label": "Exotel",
        "desc":  "Indian provider · best India rates · local +91 caller ID",
        "env":   "LIVEKIT_TRUNK_EXOTEL",
    },
}


def _trunk_id(provider: str) -> str:
    """Return the LiveKit outbound SIP trunk ID for the given provider.
    Raises HTTP 500 if the provider isn't configured."""
    meta = _TELEPHONY.get(provider)
    if meta:
        val = os.getenv(meta["env"], "").strip()
        if val:
            return val
    # Legacy fallback for Twilio
    if provider in ("twilio", ""):
        val = os.getenv("LIVEKIT_OUTBOUND_TRUNK_ID", "").strip()
        if val:
            return val
    env_hint = _TELEPHONY.get(provider, {}).get("env", "LIVEKIT_OUTBOUND_TRUNK_ID")
    raise HTTPException(
        500,
        f"No LiveKit SIP trunk configured for '{provider}'. Set {env_hint} in .env.",
    )


def _telephony_catalog() -> list[dict]:
    """All providers with a configured flag — fed to the precall UI."""
    out = []
    for key, meta in _TELEPHONY.items():
        val = os.getenv(meta["env"], "").strip()
        if not val and key == "twilio":
            val = os.getenv("LIVEKIT_OUTBOUND_TRUNK_ID", "").strip()
        out.append({
            "key":        key,
            "label":      meta["label"],
            "desc":       meta["desc"],
            "configured": bool(val),
        })
    return out


def _lk_client() -> LiveKitAPI:
    """Build a LiveKit server-side API client. Uses LIVEKIT_URL/API_KEY/SECRET."""
    return LiveKitAPI(
        url=_require("LIVEKIT_URL"),
        api_key=_require("LIVEKIT_API_KEY"),
        api_secret=_require("LIVEKIT_API_SECRET"),
    )


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "livekit_url": os.getenv("LIVEKIT_URL", ""),
        "sip_trunk_id": os.getenv("LIVEKIT_OUTBOUND_TRUNK_ID", ""),
    }


@app.get("/precall")
async def precall_page():
    """Operator-facing combo picker. JS reads call payload from ?p=<base64>."""
    return Response(
        (ROOT / "precall.html").read_text(encoding="utf-8"),
        media_type="text/html",
    )


@app.get("/api/catalog")
async def api_catalog():
    """Combos + dropdown options for the precall page."""
    return {
        "combos": [{
            "key": c.key, "label": c.label, "description": c.description,
            "stt": c.stt, "llm": c.llm, "tts": c.tts, "badge": c.badge,
        } for c in PRESETS],
        "stt": STT_OPTIONS,
        "llm": LLM_OPTIONS,
        "tts": TTS_OPTIONS,
        "telephony": _telephony_catalog(),
    }


@app.post("/api/dial")
async def api_dial(request: Request):
    """
    Place an outbound call.

    Body (JSON):
      {
        "phone":          "+91XXXXXXXXXX",
        "contact_id":     "ct-...",            # optional, threaded back via callback
        "business_id":    "biz-...",
        "contact_name":   "Asha Singh",
        "business_name":  "Acme Corp",
        "business_blurb": "We sell ...",
        "agent_name":     "Vox",
        "purpose":        "checking demo interest",
        "callback_url":   "http://localhost:8000/api/voice/callback",
        "stt": "...", "llm": "...", "tts": "...",   # individual overrides, optional
        "combo": "groq-elevenlabs"                  # OR a preset key
      }

    Returns:
      { "ok": True, "call_sid": "ca-...", "watch_url": "/calls/ca-..." }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")

    phone = (body.get("phone") or "").strip()
    if not phone or not phone.startswith("+"):
        raise HTTPException(400, "phone is required in E.164 format (e.g. +91…)")

    # Resolve stack
    stt_key = (body.get("stt") or "").strip()
    llm_key = (body.get("llm") or "").strip()
    tts_key = (body.get("tts") or "").strip()
    combo_key = (body.get("combo") or "").strip()
    if combo_key and not (stt_key and llm_key and tts_key):
        c = find_combo(combo_key) or default_combo()
        stt_key = stt_key or c.stt
        llm_key = llm_key or c.llm
        tts_key = tts_key or c.tts

    call_id = f"ca-{uuid.uuid4().hex[:14]}"
    room_name = f"call-{call_id}"

    # Job metadata — picked up by the agent worker on dispatch.
    metadata = {
        "call_id":        call_id,
        "contact_id":     body.get("contact_id", ""),
        "business_id":    body.get("business_id", ""),
        "contact_name":   (body.get("contact_name") or "there").strip(),
        "contact_phone":  phone,
        "business_name":  (body.get("business_name") or "Nexus").strip(),
        "business_blurb": (body.get("business_blurb")
                           or "We help businesses run smarter operations.").strip(),
        "agent_name":     (body.get("agent_name") or os.getenv("VOX_AGENT_NAME", "Vox")).strip(),
        "purpose":        (body.get("purpose") or "a quick check-in").strip(),
        "callback_url":   (body.get("callback_url") or "").strip(),
        "stt":            stt_key,
        "llm":            llm_key,
        "tts":            tts_key,
        "combo":          combo_key,
    }

    started_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"[api/dial] call_id={call_id} room={room_name} phone={phone} "
        f"stack={stt_key} / {llm_key} / {tts_key}"
    )

    lkapi = _lk_client()
    try:
        # 1. Dispatch agent worker into the room. The room is auto-created if
        #    it doesn't exist. metadata is JSON-serialized.
        import json as _json
        agent_name = os.getenv("LIVEKIT_AGENT_NAME", "vox")
        await lkapi.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
                metadata=_json.dumps(metadata),
            )
        )
        logger.info(f"[api/dial] dispatched agent={agent_name} → room={room_name}")

        # 2. Create the SIP outbound participant — LiveKit dials the phone
        #    through whichever provider the operator selected.
        provider = (body.get("telephony_provider") or "twilio").strip()
        trunk_id = _trunk_id(provider)
        await lkapi.sip.create_sip_participant(
            CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=phone,
                room_name=room_name,
                participant_identity=f"caller-{call_id}",
                participant_name=metadata["contact_name"],
                wait_until_answered=False,
            )
        )
        logger.info(
            f"[api/dial] SIP outbound to {phone} "
            f"via {provider} trunk={trunk_id[:10]}…"
        )
    except Exception as e:
        logger.exception(f"[api/dial] LiveKit dispatch failed: {e}")
        raise HTTPException(502, f"LiveKit dispatch failed: {e}")
    finally:
        await lkapi.aclose()

    return {
        "ok": True,
        "call_sid": call_id,
        "watch_url": f"/calls/{call_id}",
    }


# ── Cockpit (live watch) ─────────────────────────────────────────────────
@app.get("/calls/{call_id}")
async def call_cockpit(call_id: str):
    html = (ROOT / "cockpit.html").read_text(encoding="utf-8")
    return Response(html.replace("__CALL_ID__", call_id), media_type="text/html")


@app.get("/api/calls/{call_id}/summary")
async def get_call_summary(call_id: str):
    """Fallback for the cockpit when the live data-channel `summary` event
    arrived after the cockpit closed — the cockpit polls this endpoint when
    the call ends and renders the saved JSON if it exists.

    Also useful for ad-hoc inspection: paste the call_id in the URL bar."""
    import json
    from pathlib import Path
    sp = TRANSCRIPT_DIR / f"{call_id}.summary.json"
    tp = TRANSCRIPT_DIR / f"{call_id}.jsonl"
    out = {"call_id": call_id, "summary": None, "turns": []}
    if sp.exists():
        try:
            out["summary"] = json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if tp.exists():
        for line in tp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out["turns"].append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@app.get("/api/cockpit-token/{call_id}")
async def cockpit_token(call_id: str):
    """Issue a short-lived LiveKit JWT for the cockpit page. Subscribe-only —
    the viewer can't publish audio, just watch + receive data-channel events."""
    room_name = f"call-{call_id}"
    identity = f"viewer-{uuid.uuid4().hex[:8]}"
    token = (
        AccessToken(_require("LIVEKIT_API_KEY"), _require("LIVEKIT_API_SECRET"))
        .with_identity(identity)
        .with_name("cockpit viewer")
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_subscribe=True,
            can_publish=False,
            can_publish_data=False,
        ))
        .to_jwt()
    )
    return {
        "token": token,
        "url": _require("LIVEKIT_URL"),
        "room": room_name,
    }


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("VOICE_AGENT_PORT", "8765"))
    logger.info(f"NexusCaller Voice Agent on http://0.0.0.0:{port}")
    uvicorn.run("voice_agent.server:app", host="0.0.0.0", port=port, reload=False)
