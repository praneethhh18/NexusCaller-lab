"""
Post-call summary — feeds the transcript to an LLM and returns a
structured lead-gen summary the CRM can act on directly.

Provider is selected via env vars (provider-agnostic, OpenAI-compat):
  SUMMARY_PROVIDER=sambanova|groq|gemini|ollama|openai   (default: sambanova)
  SUMMARY_LLM_MODEL=Meta-Llama-3.3-70B-Instruct          (default per provider)

We default to SambaNova because Groq is Cloudflare-blocked from many
networks (notably Indian residential ISPs). SambaNova is free, fast,
and OpenAI-compatible. If the chosen provider fails, we fall back
through the chain so the CRM always gets *something* useful.

Output schema:
    {
      "outcome":              one of OUTCOMES,
      "headline":             string — one short line for the activity feed,
      "lead_score":           0-100 (CRM uses this to prioritize),
      "interest_level":       "hot" | "warm" | "cold" | "none",
      "objections":           [str, ...]    // up to 5
      "key_points":           [str, ...]    // 1-5 short factual bullets
      "key_quotes":           [str, ...]    // up to 3 verbatim caller quotes
      "next_step":            string        // imperative, single sentence
      "callback_requested_at":  ISO 8601 string OR null,
      "sentiment":            "positive" | "neutral" | "negative"
    }
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger


OUTCOMES = (
    "qualified",          # interested, agreed to next step
    "follow_up_needed",   # interested but needs a callback / more info
    "not_interested",     # answered, said no
    "wrong_number",       # not the intended person
    "voicemail",          # left a message / hit machine
    "no_answer",          # never picked up
    "call_failed",        # connection / network issue
    "unclear",            # answered but transcript ambiguous
)


SUMMARY_SYSTEM_PROMPT = f"""You are a senior sales-call analyst. Given a
transcript of an outbound phone call between an AI calling agent and a
contact, produce a JSON summary the CRM will save against the contact record.

Return JSON ONLY — no markdown fences, no preamble, no commentary.

Schema (every field required, use null/empty list when truly absent):
{{
  "outcome":               one of {" | ".join(OUTCOMES)},
  "headline":              "One short line for the activity feed (max 14 words)",
  "lead_score":            integer 0-100 (your overall confidence this lead is worth pursuing),
  "interest_level":        "hot" | "warm" | "cold" | "none",
  "objections":            ["price", "timing", "budget", "no need", ...],
  "key_points":            ["fact 1", "fact 2", ...],
  "key_quotes":            ["short verbatim 1", ...],
  "next_step":             "Imperative single sentence — what should happen next.",
  "callback_requested_at": "ISO 8601 timestamp if caller explicitly asked to be called back, else null",
  "sentiment":             "positive" | "neutral" | "negative"
}}

SCORING GUIDE (lead_score):
  90-100 = explicit purchase intent or scheduled demo
  70-89  = strong interest, asked detailed questions, agreed to follow-up
  50-69  = warm — polite, listened, didn't commit, said "maybe later"
  20-49  = cold — answered but uninterested or distracted
  0-19   = no answer / wrong number / call_failed / explicit "do not call"

INTEREST LEVEL MAPPING:
  hot  -> score 80+, said yes to next step
  warm -> score 50-79, didn't say no
  cold -> score 20-49 OR not_interested with no future hook
  none -> no_answer | voicemail | wrong_number | call_failed

CALLBACK_REQUESTED_AT:
  Set ONLY if the caller said something like "call me back tomorrow at 4pm"
  or "next Tuesday morning works." Convert to ISO 8601 in the caller's
  apparent timezone (default IST/+05:30 if not stated).
"""


# Provider configs: each maps to (base_url, api_key_env_var, default_model)
_PROVIDERS = {
    "sambanova": (
        "https://api.sambanova.ai/v1",
        "SAMBANOVA_API_KEY",
        "Meta-Llama-3.3-70B-Instruct",
    ),
    "groq": (
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.1-8b-instant",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
    "openai": (
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "gpt-4o-mini",
    ),
    "ollama": (
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        None,  # Ollama needs no key
        "llama3.1:8b-instruct-q4_K_M",
    ),
}

# Order tried when the configured provider fails. Skips ones missing keys.
_FALLBACK_CHAIN = ("sambanova", "gemini", "groq", "openai", "ollama")


def _normalize_summary(s: dict) -> dict:
    """Fill in any required fields the LLM may have skipped, so the cockpit
    + CRM never see a half-formed summary. Headline is the most important —
    it's what the cockpit and activity feed display."""
    s = dict(s)  # don't mutate caller's dict

    outcome = s.get("outcome") or "unclear"
    if outcome not in OUTCOMES:
        outcome = "unclear"
    s["outcome"] = outcome

    score = s.get("lead_score")
    try:
        s["lead_score"] = max(0, min(100, int(score)))
    except (TypeError, ValueError):
        s["lead_score"] = 0

    interest = (s.get("interest_level") or "").lower()
    if interest not in ("hot", "warm", "cold", "none"):
        score = s["lead_score"]
        interest = "hot" if score >= 80 else "warm" if score >= 50 else "cold" if score >= 20 else "none"
    s["interest_level"] = interest

    sentiment = (s.get("sentiment") or "").lower()
    if sentiment not in ("positive", "neutral", "negative"):
        sentiment = "neutral"
    s["sentiment"] = sentiment

    s["objections"]  = list(s.get("objections")  or [])[:5]
    s["key_points"]  = list(s.get("key_points")  or [])[:5]
    s["key_quotes"]  = list(s.get("key_quotes")  or [])[:3]
    s["next_step"]   = (s.get("next_step")  or "").strip() or "Review transcript and follow up if appropriate."
    s["callback_requested_at"] = s.get("callback_requested_at") or None

    # Headline is required for cockpit rendering — synthesize from outcome
    # + interest if the model didn't provide one.
    headline = (s.get("headline") or "").strip()
    if not headline:
        outcome_label = {
            "qualified":         "Qualified lead",
            "follow_up_needed":  "Follow-up needed",
            "not_interested":    "Not interested",
            "wrong_number":      "Wrong number",
            "voicemail":         "Hit voicemail",
            "no_answer":         "No answer",
            "call_failed":       "Call failed",
            "unclear":           "Inconclusive call",
        }.get(outcome, "Call completed")
        headline = f"{outcome_label} · {interest} interest · score {s['lead_score']}"
    s["headline"] = headline[:140]
    return s


def _format_transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = t.get("role", "?")
        text = (t.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _call_provider(provider: str, transcript: str) -> Optional[dict]:
    """Call a single provider; return parsed dict or None on any failure."""
    cfg = _PROVIDERS.get(provider)
    if not cfg:
        logger.warning(f"[summary] unknown provider {provider!r}")
        return None
    base_url, key_env, default_model = cfg
    api_key = os.getenv(key_env, "") if key_env else "ollama"
    if key_env and not api_key:
        logger.info(f"[summary] {provider}: no {key_env}, skipping")
        return None

    model = os.getenv("SUMMARY_LLM_MODEL", "") or default_model

    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": model,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Transcript:\n\n{transcript}"},
                    ],
                },
            )
        if r.status_code != 200:
            logger.warning(f"[summary] {provider} HTTP {r.status_code}: {r.text[:200]}")
            return None
        body = r.json()
        raw = body["choices"][0]["message"]["content"] or ""
        # Some providers wrap JSON in markdown fences even when asked not to
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:].lstrip()
            raw = raw.rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        # Backfill required fields the LLM may have skipped — cockpit
        # rendering keys on `headline` so it must always be present.
        parsed = _normalize_summary(parsed)
        logger.info(
            f"[summary] {provider} ({model}) OK · "
            f"outcome={parsed.get('outcome','?')} score={parsed.get('lead_score','?')}"
        )
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"[summary] {provider} returned non-JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"[summary] {provider} failed: {e}")
        return None


def summarise_transcript(turns: list[dict]) -> Optional[dict]:
    """Run the summary LLM and return the parsed JSON, or None on failure.
    Tries the configured provider first, then falls back through the chain
    so a flaky/blocked provider doesn't kill the CRM update."""
    if not turns:
        logger.warning("[summary] empty transcript — skipping")
        return None

    transcript = _format_transcript(turns)
    primary = (os.getenv("SUMMARY_PROVIDER") or "sambanova").lower()

    # Try primary first, then fall back through the chain (skipping primary)
    chain = [primary] + [p for p in _FALLBACK_CHAIN if p != primary]
    for provider in chain:
        result = _call_provider(provider, transcript)
        if result:
            return result

    logger.error("[summary] all providers failed — no summary generated")
    return None


def summarise_jsonl(jsonl_path: Path) -> Optional[dict]:
    """Load transcript JSONL from disk, summarise, save alongside as .summary.json."""
    if not jsonl_path.exists():
        logger.error(f"[summary] no transcript at {jsonl_path}")
        return None

    turns = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    summary = summarise_transcript(turns)
    if summary is not None:
        out = jsonl_path.with_suffix(".summary.json")
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"[summary] wrote {out}")
    return summary
