"""
Subagent Research Flow for AI Debate Engine.

When a user submits a claim or evidence request, this module runs an asynchronous 
background subagent worker thread to retrieve counter-perspectives, relevant approved 
studies, and empirical counter-evidence, then synthesizes those findings into a 
structured counter-evidence state for the agent's rebuttal.
"""

import threading
import queue
import logging
from typing import Optional, List, Dict, Any

from src.source_lookup import lookup_relevant_source_hits
from src.evidence import relevant_credible_sources, format_source_urls

logger = logging.getLogger(__name__)

# In-memory research synthesis cache by session/turn
_SUBAGENT_RESEARCH_CACHE: Dict[str, Dict[str, Any]] = {}

def spin_off_research_subagent(
    session_id: str,
    user_claim: str,
    persona_id: str,
    debate_topic: str,
    transcript: List[Dict[str, str]],
) -> None:
    """
    Spins off a background research subagent thread to fetch counter-perspectives and key studies.
    Result is stored asynchronously in _SUBAGENT_RESEARCH_CACHE for subsequent turns.
    """
    def _worker():
        try:
            logger.info("Subagent worker started for session %s, claim: %s", session_id, user_claim[:50])
            # 1. Retrieve approved source hits for counter perspectives
            source_hits = lookup_relevant_source_hits(
                persona_id=persona_id,
                debate_topic=debate_topic or user_claim,
                transcript=transcript,
                force_fallback=True,
            )

            # 2. Extract top credible sources & study cards for persona counter-arguments
            credible_cards = relevant_credible_sources(persona_id)
            counter_sources = []
            for hit in source_hits[:3]:
                counter_sources.append({
                    "title": hit.get("title", "Approved Study"),
                    "source_name": hit.get("source_name", "Academic Journal"),
                    "url": hit.get("url", ""),
                    "description": hit.get("description", "Counter-evidence analysis"),
                })
            
            if not counter_sources and credible_cards:
                for card in credible_cards[:2]:
                    counter_sources.append({
                        "title": f"Evidence Review: {card['name']}",
                        "source_name": card["name"],
                        "url": format_source_urls(card),
                        "description": f"Key research specialty: {card['specialty']}",
                    })

            # 3. Synthesize counter-perspective points
            synthesis = {
                "user_claim": user_claim,
                "counter_perspectives": counter_sources,
                "synthesized_rebuttal_points": [
                    f"Counter-evidence from {s['source_name']}: {s['title']} ({s['url']})"
                    for s in counter_sources if s.get("url")
                ]
            }

            _SUBAGENT_RESEARCH_CACHE[session_id] = synthesis
            logger.info("Subagent worker completed successfully for session %s", session_id)
        except Exception as exc:
            logger.error("Subagent worker error: %s", exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

def get_synthesized_counter_evidence(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve synthesized subagent counter-evidence for the session if available."""
    return _SUBAGENT_RESEARCH_CACHE.get(session_id)
