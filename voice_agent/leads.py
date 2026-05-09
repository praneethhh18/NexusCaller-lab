"""Lead queue — SQLite-backed list of contacts to call, plus auto-dialer state.

Lifecycle of one lead:
    pending  -> (auto-dialer picks next) -> calling
    calling  -> (call completes, summary fetched) -> completed
                                                     bucket = good | neutral | bad
    calling  -> (waited too long for summary)     -> skipped (back to pending eligible)
    failed   -> (lab dial returned an error)

Bucketing maps the LLM's outcome enum into 3 columns the operator UI shows:
    good    : qualified, follow_up_needed
    neutral : voicemail, no_answer, unclear
    bad     : not_interested, wrong_number, call_failed

Data lives at voice_agent/leads.db (SQLite). Doesn't share NexusAgent's
Postgres because the lab is a deliberately stateless calling worker — the
queue is operator-owned scratch state, not the system of record (the CRM is).
"""
from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_DB_PATH = Path(__file__).parent / "leads.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id           TEXT PRIMARY KEY,
            name         TEXT,
            phone        TEXT NOT NULL,
            notes        TEXT,
            business_id  TEXT,
            contact_id   TEXT,
            purpose      TEXT,
            status       TEXT NOT NULL DEFAULT 'pending',
            bucket       TEXT,
            outcome      TEXT,
            headline     TEXT,
            call_id      TEXT,
            created_at   TEXT NOT NULL,
            called_at    TEXT,
            completed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue_state (
            id          INTEGER PRIMARY KEY,
            running     INTEGER NOT NULL DEFAULT 0,
            settings    TEXT,
            started_at  TEXT,
            stopped_at  TEXT,
            updated_at  TEXT NOT NULL
        )
    """)
    # Singleton row for queue state — id=1
    conn.execute(
        "INSERT OR IGNORE INTO queue_state (id, running, updated_at) VALUES (1, 0, ?)",
        (_now(),),
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_bucket ON leads(bucket, completed_at DESC)")
    conn.commit()
    return conn


# ── Bucketing rules ────────────────────────────────────────────────────────
_BUCKETS: Dict[str, str] = {
    "qualified":         "good",
    "follow_up_needed":  "good",
    "not_interested":    "bad",
    "wrong_number":      "bad",
    "call_failed":       "bad",
    "voicemail":         "neutral",
    "no_answer":         "neutral",
    "unclear":           "neutral",
}


def bucket_for_outcome(outcome: Optional[str]) -> str:
    """Map an LLM outcome enum to the 3-column kanban bucket."""
    return _BUCKETS.get((outcome or "").strip().lower(), "neutral")


# ── Phone normalisation (E.164, default +91 for India) ────────────────────
def _normalize_phone(phone: str) -> str:
    if not phone:
        raise ValueError("phone is required")
    s = re.sub(r"[^\d+]", "", phone)
    if s.startswith("+") and len(s) >= 9:
        return s
    if len(s) == 10:                       # bare Indian mobile
        return "+91" + s
    if len(s) == 11 and s.startswith("0"):  # leading-zero local
        return "+91" + s[1:]
    if len(s) == 12 and s.startswith("91"):
        return "+" + s
    if len(s) >= 8:
        return "+" + s
    raise ValueError(f"phone too short or invalid: {phone!r}")


# ── CRUD ──────────────────────────────────────────────────────────────────
def add_lead(*, name: str = "", phone: str, notes: str = "",
              business_id: str = "", contact_id: str = "",
              purpose: str = "") -> Dict[str, Any]:
    phone_norm = _normalize_phone(phone)
    lid = f"ld-{uuid.uuid4().hex[:10]}"
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO leads (id, name, phone, notes, business_id, contact_id, "
            "purpose, status, created_at) VALUES (?,?,?,?,?,?,?,'pending',?)",
            (lid, (name or "").strip()[:200], phone_norm,
             (notes or "").strip()[:1000], business_id, contact_id,
             (purpose or "").strip()[:500], _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return get_lead(lid) or {}


def add_leads_bulk(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Add many leads at once. Returns counts + the IDs created. Skips rows
    that lack a phone or fail normalisation."""
    added: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for r in rows:
        try:
            out = add_lead(
                name=r.get("name", ""),
                phone=str(r.get("phone", "")),
                notes=r.get("notes", ""),
                business_id=r.get("business_id", ""),
                contact_id=r.get("contact_id", ""),
                purpose=r.get("purpose", ""),
            )
            if out:
                added.append(out["id"])
        except Exception as e:
            skipped.append({"row": r, "reason": str(e)})
    return {"added": len(added), "skipped": len(skipped),
            "ids": added, "errors": skipped[:10]}


def parse_csv(blob: str) -> List[Dict[str, Any]]:
    """Best-effort CSV → list of lead dicts. Required column: phone (any case).
    Optional: name, notes, purpose. Other columns ignored."""
    reader = csv.DictReader(io.StringIO(blob))
    out: List[Dict[str, Any]] = []
    for raw in reader:
        # Lowercase keys so 'Phone' / 'PHONE' / 'phone' all work
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not row.get("phone"):
            continue
        out.append({
            "name":    row.get("name", ""),
            "phone":   row.get("phone", ""),
            "notes":   row.get("notes", "") or row.get("note", ""),
            "purpose": row.get("purpose", "") or row.get("topic", ""),
        })
    return out


def get_lead(lead_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_leads(*, status: Optional[str] = None, bucket: Optional[str] = None,
                limit: int = 200) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM leads WHERE 1=1"
    params: list = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if bucket:
        sql += " AND bucket = ?"
        params.append(bucket)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    conn = _conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def delete_lead(lead_id: str) -> bool:
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def mark_calling(lead_id: str, call_id: str) -> None:
    conn = _conn()
    try:
        conn.execute(
            "UPDATE leads SET status = 'calling', call_id = ?, called_at = ? "
            "WHERE id = ?",
            (call_id, _now(), lead_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(lead_id: str, reason: str) -> None:
    conn = _conn()
    try:
        conn.execute(
            "UPDATE leads SET status = 'failed', headline = ?, completed_at = ?, "
            "bucket = 'bad' WHERE id = ?",
            ((reason or "")[:500], _now(), lead_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_completed(lead_id: str, *, summary: Dict[str, Any]) -> None:
    """Apply the post-call summary to the lead row."""
    outcome = (summary.get("outcome") or "").strip().lower()
    headline = (summary.get("headline") or "")[:500]
    bucket = bucket_for_outcome(outcome)
    conn = _conn()
    try:
        conn.execute(
            "UPDATE leads SET status = 'completed', outcome = ?, headline = ?, "
            "bucket = ?, completed_at = ? WHERE id = ?",
            (outcome, headline, bucket, _now(), lead_id),
        )
        conn.commit()
    finally:
        conn.close()


def reset_lead(lead_id: str) -> bool:
    """Move a completed/failed lead back to pending so it can be retried."""
    conn = _conn()
    try:
        cur = conn.execute(
            "UPDATE leads SET status = 'pending', bucket = NULL, outcome = NULL, "
            "headline = NULL, call_id = NULL, called_at = NULL, completed_at = NULL "
            "WHERE id = ?",
            (lead_id,),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def stats() -> Dict[str, int]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT status, bucket, COUNT(*) AS n FROM leads "
            "GROUP BY status, bucket"
        ).fetchall()
    finally:
        conn.close()
    out = {"pending": 0, "calling": 0, "completed": 0, "failed": 0,
           "good": 0, "neutral": 0, "bad": 0, "total": 0}
    for r in rows:
        n = r["n"]
        out["total"] += n
        if r["status"] in out:
            out[r["status"]] += n
        if r["bucket"] in out:
            out[r["bucket"]] += n
    return out


# ── Auto-dialer state ─────────────────────────────────────────────────────
def get_queue_state() -> Dict[str, Any]:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM queue_state WHERE id = 1").fetchone()
    finally:
        conn.close()
    out = dict(row) if row else {"running": 0}
    out["running"] = bool(out.get("running"))
    if out.get("settings"):
        try:
            out["settings"] = json.loads(out["settings"])
        except Exception:
            out["settings"] = {}
    else:
        out["settings"] = {}
    return out


def set_queue_state(*, running: bool, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = _now()
    conn = _conn()
    try:
        if settings is not None:
            conn.execute(
                "UPDATE queue_state SET running = ?, settings = ?, "
                "started_at = CASE WHEN ? = 1 THEN ? ELSE started_at END, "
                "stopped_at = CASE WHEN ? = 0 THEN ? ELSE stopped_at END, "
                "updated_at = ? WHERE id = 1",
                (1 if running else 0, json.dumps(settings),
                 1 if running else 0, now,
                 1 if running else 0, now, now),
            )
        else:
            conn.execute(
                "UPDATE queue_state SET running = ?, "
                "started_at = CASE WHEN ? = 1 THEN ? ELSE started_at END, "
                "stopped_at = CASE WHEN ? = 0 THEN ? ELSE stopped_at END, "
                "updated_at = ? WHERE id = 1",
                (1 if running else 0,
                 1 if running else 0, now,
                 1 if running else 0, now, now),
            )
        conn.commit()
    finally:
        conn.close()
    return get_queue_state()


def next_pending() -> Optional[Dict[str, Any]]:
    """Oldest-first pick of the next lead to call."""
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM leads WHERE status = 'pending' "
            "ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None
