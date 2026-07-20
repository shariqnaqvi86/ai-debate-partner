"""
Storage helpers: session logging and JSON export.
"""

import json
import datetime
import os

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOGFILE = os.path.join(LOGS_DIR, "logs.jsonl")
SESSIONS_DIR = os.path.join(LOGS_DIR, "sessions")


def _ensure_dirs() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def append_log(entry: dict) -> None:
    """Append a single turn entry to the JSONL log."""
    _ensure_dirs()
    with open(LOGFILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def export_session(
    session_id: str,
    persona: str,
    topic: str,
    evidence_items: list[dict],
    transcript: list[dict],
    last_scores: dict,
    last_flags: list,
) -> str:
    """
    Serialize a full session to JSON string and save to data/sessions/.
    Returns the JSON string so Streamlit can offer it as a download.
    """
    _ensure_dirs()
    payload = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "persona": persona,
        "topic": topic,
        "evidence_items": evidence_items,
        "transcript": transcript,
        "last_scores": last_scores,
        "last_flags": last_flags,
    }
    json_str = json.dumps(payload, indent=2)
    filepath = os.path.join(SESSIONS_DIR, f"session_{session_id[:8]}.json")
    with open(filepath, "w") as f:
        f.write(json_str)
    return json_str
