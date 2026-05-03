"""Filesystem layout for per-call transcript + summary artefacts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
TRANSCRIPT_DIR = ROOT / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)


def transcript_path(call_id: str) -> Path:
    return TRANSCRIPT_DIR / f"{call_id}.jsonl"


def summary_path(call_id: str) -> Path:
    return TRANSCRIPT_DIR / f"{call_id}.summary.json"
