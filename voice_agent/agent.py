"""
Vox — LiveKit Agent worker.

Runs as its own process:
    python -m voice_agent.agent dev      # development (auto-reload)
    python -m voice_agent.agent start    # production

Behaviour:
  1. Connects to LiveKit Cloud, registers as a worker.
  2. When `voice_agent.server` dispatches a job (one per outbound call), this
     worker picks it up — joins the LiveKit room, reads job metadata
     (contact + business + purpose + stack choice), builds STT/LLM/TTS
     plugins accordingly, runs the conversation.
  3. While the call is live, transcript turns are published to the room's
     data channel so the cockpit page can show them in real time.
  4. When the call ends, the worker generates a structured summary via
     Groq Llama and POSTs it back to NexusAgent's voice-callback URL so
     the CRM updates the contact record.

"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli,
)
from livekit.agents import llm as _lk_llm
from livekit.agents._exceptions import APIStatusError
from livekit.plugins import aws as lk_aws
from livekit.plugins import cartesia, deepgram, elevenlabs, openai, silero
from loguru import logger

from voice_agent.combos import default_combo, find_combo
from voice_agent.storage import transcript_path, summary_path
from voice_agent.summary import summarise_transcript


load_dotenv()


# ── Fallback LLM wrapper ─────────────────────────────────────────────────
class _FallbackLLM(_lk_llm.LLM):
    """An LLM that tries each provider in order, falling back to the next
    on rate-limit / model-deprecated / quota errors.

    Why we need this: SambaNova's free tier has tight per-minute limits
    (~5-10 RPM). One bad burst of preemptive generation calls and the
    whole call dies with `failed to generate LLM completion after
    4 attempts`. Instead, fail over to Gemini, then Groq, etc.
    """

    _RETRYABLE_STATUSES = {429, 410, 503}  # rate-limit, deprecated, unavailable

    def __init__(self, providers: list[_lk_llm.LLM]):
        super().__init__()
        if not providers:
            raise ValueError("_FallbackLLM needs at least one provider")
        self._providers = providers

    @property
    def model(self) -> str:
        return self._providers[0].model

    @property
    def provider(self) -> str:
        labels = ",".join(p.provider for p in self._providers)
        return f"fallback[{labels}]"

    def chat(self, **kwargs) -> _lk_llm.LLMStream:
        return _FallbackLLMStream(
            llm=self,
            providers=self._providers,
            retryable=self._RETRYABLE_STATUSES,
            chat_ctx=kwargs.pop("chat_ctx"),
            tools=kwargs.pop("tools", None) or [],
            conn_options=kwargs.pop("conn_options"),
            extra_kwargs=kwargs,
        )

    def prewarm(self) -> None:
        for p in self._providers:
            try:
                p.prewarm()
            except Exception:
                pass

    async def aclose(self) -> None:
        for p in self._providers:
            try:
                await p.aclose()
            except Exception:
                pass


class _FallbackLLMStream(_lk_llm.LLMStream):
    def __init__(self, *, providers, retryable, extra_kwargs, **kw):
        super().__init__(**kw)
        self._providers = providers
        self._retryable = retryable
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        last_err: Exception | None = None
        for i, prov in enumerate(self._providers):
            try:
                stream = prov.chat(
                    chat_ctx=self._chat_ctx,
                    tools=self._tools,
                    conn_options=self._conn_options,
                    **self._extra_kwargs,
                )
                async for chunk in stream:
                    self._event_ch.send_nowait(chunk)
                return  # success — done
            except APIStatusError as e:
                last_err = e
                if e.status_code in self._retryable and i < len(self._providers) - 1:
                    logger.warning(
                        f"[fallback-llm] provider {prov.provider}/{prov.model} "
                        f"returned {e.status_code} — falling back to next provider"
                    )
                    continue
                raise
            except Exception as e:
                last_err = e
                if i < len(self._providers) - 1:
                    logger.warning(
                        f"[fallback-llm] provider {prov.provider}/{prov.model} "
                        f"errored ({type(e).__name__}) — falling back"
                    )
                    continue
                raise
        if last_err:
            raise last_err


# ── Plugin builders ──────────────────────────────────────────────────────
def _build_stt(key: str, *, keyterms: list[str] | None = None):
    """Map a stack-key like 'deepgram-nova-3' or 'groq-whisper-…' to a
    LiveKit STT plugin. We bias toward names + business terms via Deepgram's
    keyterms (or OpenAI/Groq's prompt) — fixes most Indian-English mishearings."""
    if key.startswith("deepgram-"):
        model = key.removeprefix("deepgram-")
        return deepgram.STT(
            model=model,
            language="multi",          # auto-detect English / Hindi / mix
            interim_results=True,
            smart_format=True,
            keyterm=keyterms or [],
            # 150ms — compromise between snappy turns and giving soft
            # callers time to finish a thought. Final turn-end timing is
            # actually controlled by turn_handling.endpointing.min_delay
            # in AgentSession, so this just controls how soon Deepgram
            # finalizes its STT segments.
            endpointing_ms=150,
            filler_words=True,        # keep "um/uh" so STT doesn't truncate
        )
    if key.startswith("groq-whisper-"):
        model = key.removeprefix("groq-")
        prompt = ". ".join(keyterms or [])[:240] if keyterms else None
        return openai.STT(
            model=model,
            language="en",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            **({"prompt": prompt} if prompt else {}),
        )
    if key.startswith("local-whisper-"):
        # faster-whisper on CPU. Model name is the part after the prefix
        # (e.g. "tiny", "base", "small"). int8 compute keeps it usable on
        # a laptop CPU without a GPU.
        from voice_agent.local_plugins import LocalWhisperSTT
        model = key.removeprefix("local-whisper-")
        return LocalWhisperSTT(model=model)
    raise ValueError(f"Unknown STT key: {key!r}")


def _build_llm(key: str):
    if key.startswith("groq-"):
        # Groq is OpenAI-compatible — we use the openai plugin with a custom
        # base_url. Model id may itself contain a "/" (e.g.
        # "openai/gpt-oss-120b"), so just strip the leading "groq-".
        model = key.removeprefix("groq-")
        return openai.LLM(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
    if key.startswith("ollama-"):
        # Ollama exposes an OpenAI-compatible endpoint at /v1. with_ollama is
        # a thin convenience wrapper. Default base_url is http://localhost:11434/v1.
        model = key.removeprefix("ollama-")
        return openai.LLM.with_ollama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    if key.startswith("gemini-"):
        # Google Gemini via its OpenAI-compatible endpoint. Get a free key at
        # https://aistudio.google.com/apikey  — set GEMINI_API_KEY in .env.
        model = key.removeprefix("gemini-")
        return openai.LLM(
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    if key.startswith("openai-"):
        # OpenAI GPT models. Set OPENAI_API_KEY in .env.
        model = key.removeprefix("openai-")
        return openai.LLM(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if key.startswith("bedrock-"):
        # AWS Bedrock — Anthropic Claude, Amazon Nova, Meta Llama via
        # inference profiles. No use-case form needed. Pay-per-token.
        # Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_REGION in .env.
        # Model id is everything after "bedrock-", e.g.
        #   bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0
        model = key.removeprefix("bedrock-")
        return lk_aws.LLM(
            model=model,
            region=os.getenv("AWS_REGION", "us-east-1"),
            api_key=os.getenv("AWS_ACCESS_KEY_ID"),
            api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    raise ValueError(f"Unknown LLM key: {key!r}")


def _build_llm_with_fallback(primary_key: str) -> _lk_llm.LLM:
    """Build the primary LLM and tack on a fallback chain that kicks in
    on rate-limit / quota / deprecated-model errors. The chain is
    composed only of providers whose API key is present in .env so we
    don't waste a fallback slot on something that'll fail auth too."""
    primary = _build_llm(primary_key)
    chain: list[_lk_llm.LLM] = [primary]

    # Don't double-add the primary's provider in the fallback chain
    primary_prefix = primary_key.split("-", 1)[0]

    # Fallback when primary errors (rate-limit / quota / 5xx).
    # Bedrock is the most reliable cloud LLM right now — Indian
    # residential ISPs aren't blocked, AWS-grade SLAs, no daily caps.
    # Gemini second (free, high RPM), then OpenAI, then Groq.
    candidates = [
        ("AWS_SECRET_ACCESS_KEY", "bedrock-us.amazon.nova-micro-v1:0"),
        ("GEMINI_API_KEY",        "gemini-gemini-2.5-flash"),
        ("OPENAI_API_KEY",        "openai-gpt-4o-mini"),
        ("GROQ_API_KEY",          "groq-llama-3.1-8b-instant"),
    ]
    for env_key, fallback_key in candidates:
        if not os.getenv(env_key):
            continue
        if fallback_key.startswith(primary_prefix + "-"):
            continue  # don't fall back to the same provider as primary
        try:
            chain.append(_build_llm(fallback_key))
        except Exception as e:
            logger.warning(f"[fallback-llm] couldn't build {fallback_key!r}: {e}")

    if len(chain) == 1:
        return primary  # no fallback configured — return as-is

    logger.info(
        f"[fallback-llm] chain: " +
        " → ".join(f"{p.provider}/{p.model}" for p in chain)
    )
    return _FallbackLLM(chain)


def _build_tts(key: str):
    if key.startswith("deepgram-"):
        # Deepgram Aura TTS — streaming WebSocket mode. Streaming is
        # required for clean barge-in / interruption: when the user
        # interrupts, the TTS stops generating mid-stream and the audio
        # cleanly stops. With chunked HTTP, the full audio is already
        # buffered, so cutting it produces audio crackle and the agent
        # sounds like it ignored the interruption.
        # 16 kHz instead of 24 kHz default — half the resampling cost
        # for PSTN's 8 kHz μ-law output.
        model = key.removeprefix("deepgram-")
        return deepgram.TTS(
            model=model,
            sample_rate=16000,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        )
    if key.startswith("elevenlabs-"):
        model = key.removeprefix("elevenlabs-")
        return elevenlabs.TTS(
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            model=model,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
    if key.startswith("cartesia-"):
        model = key.removeprefix("cartesia-")
        return cartesia.TTS(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice=os.getenv("CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
            model=model,
        )
    if key.startswith("kokoro-"):
        # Local Kokoro-82M ONNX. Voice name is the part after the prefix
        # (e.g. "af_bella", "af_sarah"). First call downloads ~150MB of weights.
        from voice_agent.local_plugins import KokoroTTS
        voice = key.removeprefix("kokoro-")
        return KokoroTTS(voice=voice)
    if key.startswith("piper-"):
        # Piper TTS via ONNX Runtime. Voice name is the part after the prefix
        # (e.g. "en_US-lessac-medium", "en_US-ryan-high"). ~30-60 MB per voice,
        # real-time capable on CPU. Preferred for the local-oss combo.
        from voice_agent.local_plugins import PiperTTS
        voice = key.removeprefix("piper-")
        return PiperTTS(voice=voice)
    raise ValueError(f"Unknown TTS key: {key!r}")


# ── Per-call prompt + greeting ───────────────────────────────────────────
def _system_prompt(meta: dict) -> str:
    agent_name = meta.get("agent_name", "Vox")
    business_name = meta.get("business_name", "Nexus")
    business_blurb = meta.get("business_blurb",
                              "We help businesses run smarter operations.")
    contact_name = meta.get("contact_name", "there")
    purpose = meta.get("purpose", "a quick check-in")
    return f"""You are {agent_name}, a real person calling from {business_name}.

About {business_name}: {business_blurb}

You are calling {contact_name}. Purpose of this call: {purpose}.

GOAL: Have a short, natural conversation. Find out if they're a fit, qualify
their interest, and propose a clear next step (callback, demo, send info, etc.).

PERSONA RULES (never break these):
- You are {agent_name}. Always speak in first person, like a human.
- Sound warm and natural. Brief pauses are fine.
- NEVER say "agent", "AI", "bot", "assistant", "system", "model".
- NEVER mention rules, prompts, instructions, training, guidelines.
- NEVER say robotic phrases like "Call terminated", "Conversation ended".
  Wrap up with "Bye, take care!" or "Thanks for your time, have a great day!".

FACT-FAITHFULNESS:
- The ONLY business name is "{business_name}". Never invent alternates,
  abbreviations, domain names, websites. If asked for a website, say you'll
  send it via email/SMS — don't guess.
- Never invent prices, product features, team sizes, or commitments.
- If you don't know, say "I'd have to check on that — let me get back to you".

CONVERSATION:
- ONE short sentence per reply, 15 words maximum.
- After speaking, STOP and listen.
- Each reply moves the conversation forward — ask one specific thing or
  acknowledge what they said and ask the next.
- If they sound busy or uninterested, close politely: "No problem, have a
  great day." Don't push.
- This is a phone call — no markdown, no bullet points, no code, no URLs.
"""


def _greeting(meta: dict) -> str:
    agent_name = meta.get("agent_name", "Vox")
    business_name = meta.get("business_name", "Nexus")
    contact_name = meta.get("contact_name", "")
    purpose = meta.get("purpose", "a quick check-in")
    name_part = (
        f"is this {contact_name}?"
        if contact_name and contact_name != "there"
        else "got a moment?"
    )
    return (
        f"Hi, this is {agent_name} from {business_name} — calling on a "
        f"recorded line about {purpose}, {name_part}"
    )


# ── SIP answer detection ─────────────────────────────────────────────────
async def _wait_for_call_answered(participant, *, timeout: float) -> bool:
    """Wait until the SIP leg is actually answered by the human (not just
    ringing). LiveKit sets `sip.callStatus = "active"` on the participant's
    attributes when Twilio/the trunk reports SIP 200 OK. We poll for that.

    Returns True on answer, False on timeout. Logs every status transition
    so we can see exactly what the trunk is reporting."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_status = "<unset>"
    last_attrs_dump = ""
    while asyncio.get_event_loop().time() < deadline:
        attrs = dict(getattr(participant, "attributes", {}) or {})

        # Log full attributes dict the first time we see it change — helps
        # debug what the trunk is actually setting.
        attrs_dump = ",".join(f"{k}={v}" for k, v in attrs.items())
        if attrs_dump != last_attrs_dump:
            logger.info(f"[vox] participant.attributes changed: {{{attrs_dump}}}")
            last_attrs_dump = attrs_dump

        status = attrs.get("sip.callStatus") or attrs.get("callStatus") or ""
        if status != last_status:
            logger.info(f"[vox] sip.callStatus: {last_status!r} -> {status!r}")
            last_status = status
        if status == "active":
            return True
        await asyncio.sleep(0.25)

    logger.warning(f"[vox] answer timeout after {timeout}s (last status: {last_status!r})")
    return False


# ── Main entrypoint ──────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    """Called by LiveKit when a new job is dispatched to this worker."""
    await ctx.connect()

    # Job metadata is JSON we set on dispatch.
    raw = ctx.job.metadata or "{}"
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"[vox] could not parse job metadata: {raw!r}")
        meta = {}

    call_id = meta.get("call_id") or ctx.room.name
    callback_url = (meta.get("callback_url") or "").strip()

    # Resolve stack: explicit stt/llm/tts > combo preset > defaults.
    stt_key = (meta.get("stt") or "").strip()
    llm_key = (meta.get("llm") or "").strip()
    tts_key = (meta.get("tts") or "").strip()
    combo_key = (meta.get("combo") or "").strip()
    if combo_key and not (stt_key and llm_key and tts_key):
        c = find_combo(combo_key) or default_combo()
        stt_key = stt_key or c.stt
        llm_key = llm_key or c.llm
        tts_key = tts_key or c.tts

    if not (stt_key and llm_key and tts_key):
        c = default_combo()
        stt_key, llm_key, tts_key = c.stt, c.llm, c.tts

    # STT keyterm bias: contact name, business name, common sales vocab.
    contact_name = meta.get("contact_name", "")
    business_name = meta.get("business_name", "")
    keyterms = [t for t in (contact_name, business_name) if t and t != "there"]
    keyterms.extend(["BPO", "demo", "callback", "follow up", "pricing", "quote"])

    logger.info(
        f"[vox] call={call_id} stack={stt_key} / {llm_key} / {tts_key} "
        f"contact={contact_name!r}"
    )

    # Build the conversational session. min_interruption_words=1 means a
    # confident word triggers barge-in; Krisp filters echo at the audio layer.
    #
    # We use Silero VAD's default turn detection (no MultilingualModel)
    # because the latter needs ~100MB of HF model weights downloaded at
    # runtime which slowed the cold start AND crashed the worker if the
    # download failed. VAD-based turn detection works perfectly well for
    # English + Hinglish phone calls.
    # Reuse the VAD weights loaded once in prewarm() — loading per-call
    # causes a 1-2s spike + leaves Silero unable to keep up with realtime
    # input on slower CPUs. Falls back to a fresh load if prewarm was skipped.
    vad = ctx.proc.userdata.get("vad") if hasattr(ctx, "proc") else None
    if vad is None:
        vad = silero.VAD.load(
            sample_rate=8000,
            min_silence_duration=0.4,
            prefix_padding_duration=0.3,
            activation_threshold=0.30,
            deactivation_threshold=0.15,
        )

    # Use the v2-style TurnHandlingOptions API. This replaces the deprecated
    # min_endpointing_delay / allow_interruptions / min_interruption_*
    # constructor args, and unlocks `preemptive_tts` which starts TTS
    # generation BEFORE the turn is confirmed — typically ~300ms latency win
    # on top of the snappier endpointing.
    session = AgentSession(
        stt=_build_stt(stt_key, keyterms=keyterms),
        llm=_build_llm_with_fallback(llm_key),
        tts=_build_tts(tts_key),
        vad=vad,
        user_away_timeout=None,    # don't auto-end on silence (PSTN can be quiet)
        aec_warmup_duration=0.3,   # short — Twilio outbound has minimal echo
        turn_handling={
            "endpointing": {
                "mode": "fixed",
                "min_delay": 0.3,    # was 0.5 default → ~200ms faster reply
                "max_delay": 2.5,
            },
            "interruption": {
                "enabled": True,
                "mode": "adaptive",     # ML-based — fewer false positives
                "min_duration": 0.2,    # was 0.5 default → cuts off agent on shorter interrupts
                "min_words": 0,         # don't gate on word count
                "discard_audio_if_uninterruptible": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            },
            "preemptive_generation": {
                # Preemptive LLM (warms up the model on partial transcripts)
                # is still on — keeps latency low. But preemptive_tts and
                # high retry counts caused us to burn through SambaNova's
                # free-tier RPM (~5-10 req/min) by firing 3-4 LLM calls
                # per user turn.
                "enabled": True,
                "preemptive_tts": False,   # don't pre-generate audio — too many LLM calls
                "max_retries": 1,          # one preemptive attempt per turn (not 3)
                "max_speech_duration": 10.0,
            },
        },
    )

    # Transcript collector → also broadcasts to data-channel for cockpit viewers.
    turns: list[dict] = []
    started_at_iso = datetime.now(timezone.utc).isoformat()
    started_perf = asyncio.get_event_loop().time()

    def _record_and_broadcast(role: str, text: str):
        text = (text or "").strip()
        if not text:
            return
        turn = {
            "role": role,
            "text": text,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        turns.append(turn)
        # Persist to disk as we go (lab keeps the JSONL alongside the call).
        with transcript_path(call_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn) + "\n")
        logger.info(f"[transcript:{call_id[:8]}] {role}: {text[:80]}")
        # Push to live cockpit watchers.
        asyncio.create_task(
            ctx.room.local_participant.publish_data(
                json.dumps({"type": "turn", **turn}).encode("utf-8"),
                reliable=True,
            )
        )

    @session.on("user_input_transcribed")
    def _on_user(event):
        if getattr(event, "is_final", True):
            _record_and_broadcast("user", event.transcript)

    @session.on("conversation_item_added")
    def _on_item(event):
        # `event.item` can be a chat message OR a control object like
        # AgentHandoff — only chat messages have role/text_content.
        item = event.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role == "assistant" and text:
            _record_and_broadcast("assistant", text)

    # Push initial state event so the cockpit shows "live".
    asyncio.create_task(
        ctx.room.local_participant.publish_data(
            json.dumps({
                "type": "state", "state": "connected",
                "agent_name":     meta.get("agent_name", "Vox"),
                "contact_name":   meta.get("contact_name", ""),
                "business_name":  meta.get("business_name", ""),
                "ts":             started_at_iso,
            }).encode("utf-8"),
            reliable=True,
        )
    )

    agent = Agent(instructions=_system_prompt(meta))
    await session.start(agent=agent, room=ctx.room)

    # Greet AFTER the SIP participant actually joins the room AND the audio
    # path has had a moment to settle. wait_for_participant() fires the
    # instant the SIP leg is established, but on PSTN there's a real gap
    # between "answered" and "audio is actually flowing both ways" — on
    # Indian mobile carriers it can be 1-2 seconds.  If we speak too soon,
    # the caller still has the phone moving toward their ear and misses
    # the start of the greeting.
    # Two-phase wait:
    #
    #   1. wait_for_participant() — SIP leg established, participant joins
    #      room. The phone is now RINGING but caller hasn't picked up yet.
    #
    #   2. _wait_for_call_answered() — wait for sip.callStatus == "active"
    #      (set by LiveKit when Twilio reports SIP 200 OK = answered).
    #      Without this, we'd greet into a ringing phone and the caller
    #      hears nothing when they finally pick up.
    logger.info(f"[vox] waiting for SIP participant to join room {ctx.room.name}")
    sip_participant = None
    try:
        sip_participant = await ctx.wait_for_participant()
        logger.info(f"[vox] SIP leg established (identity={sip_participant.identity}) — waiting for caller to pick up…")
    except Exception as e:
        logger.warning(f"[vox] wait_for_participant failed (continuing anyway): {e}")

    if sip_participant is not None:
        # Subscribe to attribute changes BEFORE polling, so we don't miss
        # the moment sip.callStatus flips to "active" between polls.
        answered_event = asyncio.Event()

        @ctx.room.on("participant_attributes_changed")
        def _on_attrs_changed(changed_attrs, p):
            if p.identity != sip_participant.identity:
                return
            logger.info(f"[vox] caller attributes changed: {dict(changed_attrs)}")
            status = changed_attrs.get("sip.callStatus", "")
            if status == "active":
                answered_event.set()

        # Also check current state in case "active" was already set
        try:
            cur = dict(getattr(sip_participant, "attributes", {}) or {})
            logger.info(f"[vox] caller initial attributes: {cur}")
            if cur.get("sip.callStatus") == "active":
                answered_event.set()
        except Exception:
            pass

        try:
            await asyncio.wait_for(answered_event.wait(), timeout=60.0)
            logger.info("[vox] caller answered — letting audio path settle…")
        except asyncio.TimeoutError:
            logger.warning("[vox] no answer within 60s — speaking anyway (voicemail or stuck dial)")

    # Settling pause: 0.5s after pickup so the caller has the phone fully
    # to their ear and the carrier audio path is bidirectional.
    await asyncio.sleep(0.5)
    logger.info("[vox] speaking greeting")
    await session.say(_greeting(meta))

    # Re-broadcast the connected state now that the caller is on the line.
    # The earlier broadcast may have been lost if no cockpit was subscribed
    # yet; re-sending here makes sure late-joining viewers see "Live".
    asyncio.create_task(
        ctx.room.local_participant.publish_data(
            json.dumps({
                "type": "state", "state": "active",
                "agent_name":     meta.get("agent_name", "Vox"),
                "contact_name":   meta.get("contact_name", ""),
                "business_name":  meta.get("business_name", ""),
                "ts": datetime.now(timezone.utc).isoformat(),
            }).encode("utf-8"),
            reliable=True,
        )
    )

    # Register cleanup: when the call ends, save summary + post callback.
    # Built defensively — every step is independent and logs its own outcome
    # so a failure in one step (e.g. cockpit gone, Groq timeout) doesn't
    # block the next.  Without this, a half-failed shutdown means the CRM
    # never sees the call.
    async def _on_shutdown():
        elapsed = int(asyncio.get_event_loop().time() - started_perf)
        ended_at_iso = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"[vox] SHUTDOWN call={call_id} elapsed={elapsed}s "
            f"turns={len(turns)} callback={'yes' if callback_url else 'no'}"
        )

        # Step 1 — push "ended" state to any live cockpit watchers.
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({
                    "type": "state", "state": "ended",
                    "duration_sec": elapsed, "ts": ended_at_iso,
                }).encode("utf-8"),
                reliable=True,
            )
            logger.info("[vox] shutdown step 1/4: ended-state pushed to cockpit")
        except Exception as e:
            logger.warning(f"[vox] shutdown step 1/4: cockpit push failed (room gone): {e}")

        # Step 2 — generate the lead-gen summary via Groq.  Off the event
        # loop because the underlying HTTP call is sync.
        summary = None
        try:
            summary = await asyncio.to_thread(summarise_transcript, turns)
            if summary:
                logger.info(
                    f"[vox] shutdown step 2/4: summary generated "
                    f"headline={summary.get('headline','?')!r} "
                    f"score={summary.get('lead_score','?')} "
                    f"outcome={summary.get('outcome','?')}"
                )
            else:
                logger.warning("[vox] shutdown step 2/4: summary returned None (empty transcript?)")
        except Exception as e:
            logger.exception(f"[vox] shutdown step 2/4: summary generation crashed: {e}")

        # Step 3 — persist summary to disk regardless of whether we have
        # the live cockpit available.  Cockpit can fetch via /api/calls/{id}/summary
        # as a fallback.
        try:
            summary_path(call_id).write_text(
                json.dumps(summary or {"outcome": "unclear",
                                       "headline": "Summary unavailable",
                                       "lead_score": 0,
                                       "interest_level": "none",
                                       "sentiment": "neutral",
                                       "key_points": [],
                                       "next_step": "",
                                       "callback_requested_at": None,
                                       "objections": [],
                                       "key_quotes": []},
                           indent=2),
                encoding="utf-8",
            )
            logger.info(f"[vox] shutdown step 3/4: summary written → {summary_path(call_id)}")
        except Exception as e:
            logger.exception(f"[vox] shutdown step 3/4: summary save failed: {e}")

        # Best-effort push of summary to cockpit (will silently fail if
        # cockpit already closed — that's why the disk save in step 3 + the
        # /api/calls/{id}/summary endpoint exist).
        if summary:
            try:
                await ctx.room.local_participant.publish_data(
                    json.dumps({"type": "summary", "summary": summary,
                                "ts": ended_at_iso}).encode("utf-8"),
                    reliable=True,
                )
            except Exception:
                pass

        # Step 4 — POST callback to NexusAgent so the CRM contact record
        # gets updated.  This is the most important side-effect of the
        # shutdown — without it the call doesn't show up in the contact's
        # Vox-calls history.
        if callback_url:
            try:
                await _post_callback(callback_url, {
                    "call_sid":      call_id,
                    "contact_id":    meta.get("contact_id", ""),
                    "business_id":   meta.get("business_id", ""),
                    "contact_name":  meta.get("contact_name", ""),
                    "contact_phone": meta.get("contact_phone", ""),
                    "started_at":    started_at_iso,
                    "ended_at":      ended_at_iso,
                    "duration_sec":  elapsed,
                    "turns":         turns,
                    "summary":       summary or {},
                    "watch_url":     f"/calls/{call_id}",
                })
                logger.info("[vox] shutdown step 4/4: callback POSTed to NexusAgent")
            except Exception as e:
                logger.exception(f"[vox] shutdown step 4/4: callback failed: {e}")
        else:
            logger.info("[vox] shutdown step 4/4: no callback_url, skipping CRM update")

    ctx.add_shutdown_callback(_on_shutdown)

    # When the caller hangs up, end the call promptly. Without this, the
    # agent keeps the room open until LiveKit's idle timeout fires — which
    # can take 30-60s. ctx.shutdown() is SYNCHRONOUS in livekit-agents
    # 1.5.x — it just signals the worker to start shutdown; the registered
    # _on_shutdown coroutine runs from the worker's main loop.
    @ctx.room.on("participant_disconnected")
    def _on_caller_disconnect(participant):
        identity = getattr(participant, "identity", "")
        if not identity.startswith("caller-"):
            return
        logger.info(f"[vox] caller disconnected (identity={identity}) — ending call")
        try:
            ctx.shutdown(reason="caller_hangup")
        except Exception as e:
            logger.warning(f"[vox] ctx.shutdown() failed: {e}")


async def _post_callback(url: str, payload: dict, *, attempts: int = 3) -> None:
    """Best-effort POST to NexusAgent's /api/voice/callback."""
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("VOICE_CALLBACK_SECRET", "")
    if secret:
        headers["X-Voice-Callback-Secret"] = secret
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(url, json=payload, headers=headers)
            if 200 <= r.status_code < 300:
                logger.info(f"[callback] POST {url} → {r.status_code}")
                return
            logger.warning(f"[callback] attempt {attempt} → HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"[callback] attempt {attempt} failed: {e}")
        await asyncio.sleep(2 * attempt)
    logger.error(f"[callback] all {attempts} attempts failed for {url}")


def prewarm(proc):
    """
    Runs once per worker process at boot, BEFORE any job is dispatched.
    We use it to (a) load Silero VAD weights into shared memory and
    (b) trigger the local-OSS model downloads + ONNX session warmup so
    the very first call doesn't sit in dead silence while ~150 MB of
    Kokoro weights stream from GitHub.

    Pre-warm runs in a separate process from the call entrypoint, so
    "loaded" here means "files are on disk". The actual ONNX session is
    re-instantiated in the call process — but that's a 2-3 s init from
    a local file, not a 60+ s download.
    """
    # 8 kHz matches the PSTN audio rate (skips upsampling) and roughly
    # halves Silero's per-frame inference cost — fixes "inference is
    # slower than realtime" warnings on slower CPUs.
    #
    # Tuned for phone-to-ear callers who speak softly:
    #   activation_threshold=0.30  (default 0.5) → picks up quieter speech
    #   deactivation_threshold=0.15 (vs auto 0.35) → keeps speech latched
    #     so a soft trailing word isn't cut off mid-thought
    #   prefix_padding_duration=0.3 → less front-clipping on quick replies
    proc.userdata["vad"] = silero.VAD.load(
        sample_rate=8000,
        min_speech_duration=0.05,
        min_silence_duration=0.4,
        prefix_padding_duration=0.3,
        activation_threshold=0.30,
        deactivation_threshold=0.15,
    )

    if os.getenv("LOCAL_OSS_PREWARM", "1") != "0":
        try:
            from voice_agent.local_plugins import (
                LocalWhisperSTT, PiperTTS, _ensure_piper_voice,
            )
            logger.info("[prewarm] downloading Piper voice files (one-time, ~30MB)…")
            _ensure_piper_voice("en_US-lessac-medium")
            logger.info("[prewarm] loading Piper ONNX session…")
            piper = PiperTTS(voice="en_US-lessac-medium")
            piper._ensure_model()
            logger.info("[prewarm] loading faster-whisper tiny weights…")
            stt = LocalWhisperSTT(model="tiny")
            stt._ensure_model()
            logger.info("[prewarm] local OSS stack ready")
        except Exception as e:
            # Don't kill the worker if local-stack prewarm fails —
            # cloud combos still work without these models.
            logger.warning(f"[prewarm] local OSS prewarm skipped: {e}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=os.getenv("LIVEKIT_AGENT_NAME", "vox"),
    ))
