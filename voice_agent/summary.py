"""
Post-call summary — feeds the transcript to Groq Llama and returns a
structured lead-gen summary the CRM can act on directly.

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

from groq import Groq
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
  "objections":            ["price", "timing", "budget", "no need", ...],   // 0-5 items, lowercase short tags
  "key_points":            ["fact 1", "fact 2", ...],                       // 1-5 factual bullets
  "key_quotes":            ["short verbatim 1", ...],                       // 0-3 direct caller quotes
  "next_step":             "Imperative single sentence — what should happen next, e.g. 'Send pricing sheet by EOD Friday.'",
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
  hot  → score 80+, said yes to next step
  warm → score 50-79, didn't say no
  cold → score 20-49 OR not_interested with no future hook
  none → no_answer | voicemail | wrong_number | call_failed

CALLBACK_REQUESTED_AT:
  Set ONLY if the caller said something like "call me back tomorrow at 4pm"
  or "next Tuesday morning works." Convert to ISO 8601 in the caller's
  apparent timezone (default IST/+05:30 if not stated).
"""


def _format_transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = t.get("role", "?")
        text = (t.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def summarise_transcript(turns: list[dict]) -> Optional[dict]:
    """Run the summary LLM and return the parsed JSON, or None on failure."""
    if not turns:
        logger.warning("[summary] empty transcript — skipping")
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("[summary] GROQ_API_KEY missing")
        return None

    client = Groq(api_key=api_key)
    body = _format_transcript(turns)

    try:
        resp = client.chat.completions.create(
            model=os.getenv("SUMMARY_LLM", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n\n{body}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[summary] non-JSON: {e}\nraw={raw[:200]}")
        return None
    except Exception as e:
        logger.error(f"[summary] Groq call failed: {e}")
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
