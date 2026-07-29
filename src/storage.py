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


def export_session_markdown(
    session_id: str,
    persona: str,
    topic: str,
    transcript: list[dict],
    scores: dict,
    flags: list[dict] = None,
    rewrites: list[str] = None,
) -> str:
    """Export transcript and coaching feedback as a human-readable Markdown file."""
    lines = [
        "# AI Debate Session Report",
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Session ID:** `{session_id}`",
        f"**AI Persona:** {persona}",
        f"**Debate Topic:** {topic or 'General Policy'}",
        "\n---\n",
        "## 📊 Persuasion Scores",
        f"- **Ethos (Credibility):** {scores.get('ethos', 0)} / 100",
        f"- **Logos (Logic & Evidence):** {scores.get('logos', 0)} / 100",
        f"- **Pathos (Persuasion & Framing):** {scores.get('pathos', 0)} / 100",
        "\n---\n",
        "## 💬 Debate Transcript\n",
    ]
    for turn in transcript:
        speaker = "Student" if turn["role"] == "user" else f"AI ({persona})"
        lines.append(f"### {speaker}\n{turn['content']}\n")

    if rewrites:
        lines.append("---\n## 💡 Coaching Rewrites\n")
        for i, rw in enumerate(rewrites, 1):
            lines.append(f"{i}. {rw}")

    return "\n".join(lines)
