"""
Function-calling tools for Vox.

These are the actions the LLM can take mid-conversation. Each tool calls
NexusAgent's HTTP API to do the actual work — the voice agent itself is
stateless about CRM data.

Tools:
  - lookup_business_info(query)            search the business KB via RAG
  - schedule_callback(when_iso, reason)    create a CRM callback task
  - send_email_followup(subject, body)     queue an email (human reviews)
  - end_call(reason)                       gracefully end the call

The tools are bound to the current call's metadata (business_id,
contact_id, call_sid) via a closure built per-call in agent.py.
"""
from __future__ import annotations

import os
from typing import Annotated

import httpx
from livekit.agents import RunContext, function_tool
from loguru import logger


def _nexus_url(path: str) -> str:
    base = os.getenv("NEXUSAGENT_BASE_URL", "http://localhost:8000").rstrip("/")
    return f"{base}{path}"


def _nexus_headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    secret = os.getenv("VOICE_CALLBACK_SECRET", "")
    if secret:
        h["X-Voice-Callback-Secret"] = secret
    return h


def build_tools(*, business_id: str, contact_id: str, call_sid: str) -> list:
    """Return the list of @function_tool callables bound to this call's
    metadata. Called once per job from agent.py:entrypoint()."""

    @function_tool
    async def lookup_business_info(
        ctx: RunContext,
        query: Annotated[str, "What to look up — e.g. 'pricing for the Pro plan' or 'office hours'"],
    ) -> str:
        """Search the business knowledge base for facts about products,
        pricing, FAQs, policies. Use this BEFORE making any factual
        claim about the business. Returns a short text summary you can
        quote back to the caller naturally."""
        logger.info(f"[tool:lookup_business_info] query={query!r}")
        # Speak a filler phrase IMMEDIATELY so the caller hears we're
        # working — avoids the dead-air feeling while we round-trip to
        # the RAG backend. Picked from a short, natural set so it doesn't
        # sound robotic on repeated calls.
        try:
            import random as _r
            await ctx.session.say(
                _r.choice([
                    "Let me check that for you.",
                    "One sec, looking that up.",
                    "Hmm, give me a moment.",
                ]),
                allow_interruptions=True,
            )
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.post(
                    _nexus_url("/api/voice/agent/rag-query"),
                    headers=_nexus_headers(),
                    json={"business_id": business_id, "query": query, "top_k": 3},
                )
            if r.status_code != 200:
                logger.warning(f"[tool:lookup_business_info] HTTP {r.status_code}: {r.text[:120]}")
                return "I don't have that detail handy — I'll follow up by email."
            data = r.json()
            formatted = (data.get("formatted") or "").strip()
            if not formatted or formatted.startswith("(no"):
                return "I don't have that detail handy — I'll follow up by email."
            return formatted[:1200]
        except Exception as e:
            logger.warning(f"[tool:lookup_business_info] failed: {e}")
            return "I don't have that detail right now — I can email it to you."

    @function_tool
    async def schedule_callback(
        ctx: RunContext,
        when_iso: Annotated[str, "ISO 8601 datetime, e.g. '2026-05-09T15:30:00+05:30'. Use IST (+05:30) by default."],
        reason: Annotated[str, "Short reason — e.g. 'caller wants product demo' or 'call back after lunch'"],
    ) -> str:
        """Schedule a callback task in the CRM. Use this when the caller
        asks to be called back at a specific time, or when they say
        'busy now, try later'. The CRM will create a task assigned to
        the right person. Confirm the time back to the caller after
        calling this tool."""
        logger.info(f"[tool:schedule_callback] when={when_iso} reason={reason!r}")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.post(
                    _nexus_url("/api/voice/agent/schedule-callback"),
                    headers=_nexus_headers(),
                    json={
                        "business_id": business_id,
                        "contact_id":  contact_id,
                        "when_iso":    when_iso,
                        "reason":      reason,
                        "call_sid":    call_sid,
                    },
                )
            if r.status_code != 200:
                logger.warning(f"[tool:schedule_callback] HTTP {r.status_code}: {r.text[:120]}")
                return "I had trouble scheduling that — let me note it down and our team will reach out."
            return f"Callback scheduled for {when_iso}."
        except Exception as e:
            logger.warning(f"[tool:schedule_callback] failed: {e}")
            return "I had trouble scheduling that — our team will reach out."

    @function_tool
    async def send_email_followup(
        ctx: RunContext,
        subject: Annotated[str, "Subject line, e.g. 'Pricing for Pro plan' or 'Demo materials as requested'"],
        body: Annotated[str, "Email body in plain text. Be concise; mention what was discussed on the call."],
    ) -> str:
        """Queue an email to the contact with the requested information
        (pricing, demo links, brochures, etc.). Use this when the caller
        asks you to send something. The email is queued for human review
        first — don't promise instant delivery. Confirm to the caller
        that the email will go out shortly."""
        logger.info(f"[tool:send_email_followup] subject={subject!r}")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.post(
                    _nexus_url("/api/voice/agent/send-email"),
                    headers=_nexus_headers(),
                    json={
                        "business_id": business_id,
                        "contact_id":  contact_id,
                        "subject":     subject,
                        "body":        body,
                        "call_sid":    call_sid,
                    },
                )
            if r.status_code != 200:
                logger.warning(f"[tool:send_email_followup] HTTP {r.status_code}: {r.text[:120]}")
                return "I'll have the team send that to you shortly."
            return f"Email queued: '{subject}'. Will go out shortly."
        except Exception as e:
            logger.warning(f"[tool:send_email_followup] failed: {e}")
            return "I'll have the team send that to you shortly."

    @function_tool
    async def end_call(
        ctx: RunContext,
        reason: Annotated[str, "Brief reason — e.g. 'caller said bye', 'caller not interested', 'goal achieved'"],
    ) -> str:
        """End the call gracefully. Use ONLY after you've said goodbye
        out loud — this hangs up the phone. Don't call it mid-sentence.
        Don't call it if the caller might still want to talk."""
        logger.info(f"[tool:end_call] reason={reason!r}")
        # The shutdown happens via the JobContext; we trigger it through
        # the session's room. The caller should already have heard the
        # goodbye — this just terminates the audio leg.
        try:
            session = ctx.session
            room = getattr(session, "_room", None)
            # Disconnect the SIP participant which triggers our shutdown
            # listener in agent.py (which calls ctx.shutdown()).
            if room is not None:
                await session.aclose()  # cleanly close the session
        except Exception as e:
            logger.warning(f"[tool:end_call] graceful close failed: {e}")
        return "Call ended."

    return [lookup_business_info, schedule_callback, send_email_followup, end_call]
