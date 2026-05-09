"""Lead-queue REST router + background auto-dialer.

Mounted by server.py at boot. Adds:
    GET  /leads                        operator UI (kanban: pending / good / bad)
    POST /api/leads                    add one lead
    POST /api/leads/bulk               add many (JSON list OR CSV upload)
    GET  /api/leads                    list (filter by status, bucket)
    GET  /api/leads/stats              counts for the kanban headers
    DELETE /api/leads/{id}             remove a lead
    POST /api/leads/{id}/reset         retry a failed/completed lead
    POST /api/leads/{id}/call          dial this specific lead now (one-off)
    GET  /api/leads/queue/status       running? settings? next?
    POST /api/leads/queue/start        start the auto-dialer
    POST /api/leads/queue/stop         stop the auto-dialer

The auto-dialer is a single asyncio task spawned by server.py at boot.
It loops until stopped:
    1. Pick next pending lead
    2. Dial via the SAME pipeline /api/dial uses (so all callbacks +
       transcripts continue to flow back to NexusAgent unchanged)
    3. Wait for completion: poll for the {call_id}.summary.json file
       (written by the agent worker on session shutdown). Bounded wait.
    4. Mark lead completed + bucketed
    5. Sleep `delay_sec` between calls
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile, File
from loguru import logger

from voice_agent import leads as _leads
from voice_agent import storage as _storage


router = APIRouter()
ROOT = Path(__file__).parent

# Cap on how long the auto-dialer waits for a single call to complete before
# giving up and moving to the next lead. Real calls usually finish within 90s
# (no answer + voicemail + summary write); a generous ceiling avoids stalling
# the queue if a summary file never lands.
_MAX_CALL_WAIT_SEC = 240
_POLL_INTERVAL_SEC = 4

# Default delay between calls — gives operators time to listen if they're
# tailing the cockpit. Tunable via queue settings.
_DEFAULT_DELAY_SEC = 12


# ── UI page ────────────────────────────────────────────────────────────────
@router.get("/leads")
async def leads_page():
    return Response(
        (ROOT / "leads.html").read_text(encoding="utf-8"),
        media_type="text/html",
    )


# ── CRUD ──────────────────────────────────────────────────────────────────
@router.post("/api/leads")
async def api_add_lead(request: Request):
    body = await request.json()
    if not body.get("phone"):
        raise HTTPException(400, "phone is required")
    try:
        out = _leads.add_lead(
            name=body.get("name", ""),
            phone=body["phone"],
            notes=body.get("notes", ""),
            business_id=body.get("business_id", ""),
            contact_id=body.get("contact_id", ""),
            purpose=body.get("purpose", ""),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return out


@router.post("/api/leads/bulk")
async def api_add_bulk(request: Request,
                        file: Optional[UploadFile] = File(None)):
    """Two paths:
        - multipart/form-data with `file=<csv>` → parse the CSV
        - application/json with `{"leads": [...]}` → add the list as-is
    Required column / field: `phone`. Optional: name, notes, purpose.
    """
    if file is not None:
        blob = (await file.read()).decode("utf-8", errors="ignore")
        rows = _leads.parse_csv(blob)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Send a JSON body with `leads:[...]` or upload a CSV file.")
        rows = body.get("leads") or []
        if not isinstance(rows, list):
            raise HTTPException(400, "`leads` must be a list of objects.")
    if not rows:
        raise HTTPException(400, "No valid rows found.")
    return _leads.add_leads_bulk(rows)


@router.get("/api/leads")
async def api_list_leads(status: str = "", bucket: str = "", limit: int = 200):
    return {
        "leads": _leads.list_leads(
            status=status or None, bucket=bucket or None, limit=limit
        ),
    }


@router.get("/api/leads/stats")
async def api_leads_stats():
    return _leads.stats()


@router.delete("/api/leads/{lead_id}")
async def api_delete_lead(lead_id: str):
    if not _leads.delete_lead(lead_id):
        raise HTTPException(404, "lead not found")
    return {"ok": True}


@router.post("/api/leads/{lead_id}/reset")
async def api_reset_lead(lead_id: str):
    if not _leads.reset_lead(lead_id):
        raise HTTPException(404, "lead not found")
    return _leads.get_lead(lead_id)


# ── One-off manual dial ───────────────────────────────────────────────────
@router.post("/api/leads/{lead_id}/call")
async def api_call_lead(lead_id: str, request: Request):
    """Dial this lead now without going through the queue. Useful when the
    operator wants to test a single lead before starting auto-dial."""
    lead = _leads.get_lead(lead_id)
    if not lead:
        raise HTTPException(404, "lead not found")
    if lead["status"] == "calling":
        raise HTTPException(409, "already being called")

    # Pull stack settings from the request body if present, else from queue defaults
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if not body:
        qs = _leads.get_queue_state().get("settings") or {}
        body = dict(qs)

    out = await _dial_one_lead(lead, body)
    return out


# ── Queue control ──────────────────────────────────────────────────────────
@router.get("/api/leads/queue/status")
async def api_queue_status():
    state = _leads.get_queue_state()
    nxt = _leads.next_pending()
    state["next"] = {
        "id":    nxt["id"],
        "name":  nxt["name"],
        "phone": nxt["phone"],
    } if nxt else None
    state["stats"] = _leads.stats()
    return state


@router.post("/api/leads/queue/start")
async def api_queue_start(request: Request):
    """Start the auto-dialer. Body (all optional):
       {
         "combo": "groq-elevenlabs",
         "telephony_provider": "twilio",
         "delay_sec": 12,
         "callback_url": "http://localhost:8000/api/voice/callback",
         "business_id":  "biz-...",
         "business_name": "...",
         "business_blurb": "...",
       }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    settings = {
        "combo":              (body.get("combo") or "").strip(),
        "stt":                (body.get("stt") or "").strip(),
        "llm":                (body.get("llm") or "").strip(),
        "tts":                (body.get("tts") or "").strip(),
        "telephony_provider": (body.get("telephony_provider") or "twilio").strip(),
        "delay_sec":          int(body.get("delay_sec") or _DEFAULT_DELAY_SEC),
        "callback_url":       (body.get("callback_url") or "").strip(),
        "business_id":        (body.get("business_id") or "").strip(),
        "business_name":      (body.get("business_name") or "").strip(),
        "business_blurb":     (body.get("business_blurb") or "").strip(),
        "agent_name":         (body.get("agent_name") or "Vox").strip(),
    }
    return _leads.set_queue_state(running=True, settings=settings)


@router.post("/api/leads/queue/stop")
async def api_queue_stop():
    return _leads.set_queue_state(running=False)


# ── Auto-dialer loop ───────────────────────────────────────────────────────
async def _dial_one_lead(lead: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Dial one lead by reusing /api/dial's pipeline programmatically.

    We build the same JSON body /api/dial accepts, then POST to it via the
    in-process app — that way a single change to /api/dial benefits both
    operator-driven and queue-driven calls.
    """
    from voice_agent.server import app  # late import to avoid circular ref
    import httpx

    body = {
        "phone":            lead["phone"],
        "contact_id":       lead.get("contact_id") or "",
        "business_id":      settings.get("business_id") or lead.get("business_id") or "",
        "contact_name":     lead.get("name") or "there",
        "business_name":    settings.get("business_name") or "",
        "business_blurb":   settings.get("business_blurb") or "",
        "agent_name":       settings.get("agent_name") or "Vox",
        "purpose":          lead.get("purpose") or "a quick check-in",
        "callback_url":     settings.get("callback_url") or "",
        "stt":              settings.get("stt") or "",
        "llm":              settings.get("llm") or "",
        "tts":              settings.get("tts") or "",
        "combo":            settings.get("combo") or "",
        "telephony_provider": settings.get("telephony_provider") or "twilio",
    }

    # Use ASGI transport so we hit /api/dial in-process — no real HTTP hop.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                  base_url="http://lab.local") as client:
        resp = await client.post("/api/dial", json=body, timeout=30.0)

    if resp.status_code != 200:
        reason = f"HTTP {resp.status_code}: {resp.text[:200]}"
        _leads.mark_failed(lead["id"], reason)
        return {"ok": False, "reason": reason}

    data = resp.json()
    call_id = data.get("call_sid")
    if not call_id:
        _leads.mark_failed(lead["id"], "lab returned no call_sid")
        return {"ok": False, "reason": "no call_sid"}

    _leads.mark_calling(lead["id"], call_id)
    return {"ok": True, "call_sid": call_id, "watch_url": data.get("watch_url")}


async def _wait_for_completion(call_id: str) -> Optional[Dict[str, Any]]:
    """Poll the agent's summary file. Returns the parsed summary or None on timeout."""
    sp = _storage.summary_path(call_id)
    waited = 0
    while waited < _MAX_CALL_WAIT_SEC:
        if sp.exists():
            try:
                return json.loads(sp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # File is being written; wait one more cycle
                pass
        await asyncio.sleep(_POLL_INTERVAL_SEC)
        waited += _POLL_INTERVAL_SEC
    return None


async def _autodialer_loop():
    """Single forever-task spawned by server.py at boot. Polls the queue state."""
    logger.info("[leads] auto-dialer task started")
    while True:
        try:
            state = _leads.get_queue_state()
            if not state.get("running"):
                # Queue paused — light sleep and re-check
                await asyncio.sleep(2)
                continue

            lead = _leads.next_pending()
            if not lead:
                # Empty queue — pause naturally and re-check
                await asyncio.sleep(5)
                continue

            settings = state.get("settings") or {}
            logger.info(f"[leads] dialing next: {lead['id']} {lead['phone']}")
            dial = await _dial_one_lead(lead, settings)
            if not dial.get("ok"):
                logger.warning(f"[leads] dial failed for {lead['id']}: {dial.get('reason')}")
                # Don't re-pick the failed lead in the next loop — already marked failed
                await asyncio.sleep(int(settings.get("delay_sec") or _DEFAULT_DELAY_SEC))
                continue

            summary = await _wait_for_completion(dial["call_sid"])
            if summary:
                _leads.mark_completed(lead["id"], summary=summary)
                logger.info(f"[leads] {lead['id']} completed → {summary.get('outcome')}")
            else:
                _leads.mark_failed(lead["id"], "no summary within timeout")
                logger.warning(f"[leads] {lead['id']} timed out waiting for summary")

            await asyncio.sleep(int(settings.get("delay_sec") or _DEFAULT_DELAY_SEC))

        except asyncio.CancelledError:
            logger.info("[leads] auto-dialer task cancelled")
            raise
        except Exception as e:
            logger.exception(f"[leads] auto-dialer iteration crashed: {e}")
            await asyncio.sleep(5)


def install_autodialer(app):
    """Wire the auto-dialer into FastAPI's startup/shutdown hooks."""
    @app.on_event("startup")
    async def _start_autodialer():
        app.state._autodialer_task = asyncio.create_task(_autodialer_loop())

    @app.on_event("shutdown")
    async def _stop_autodialer():
        task = getattr(app.state, "_autodialer_task", None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
