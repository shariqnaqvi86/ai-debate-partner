"""
Debate prompt builder, topic inference, and transcript summariser.
"""

import hashlib
import re
from typing import Optional

from src.evidence import classify_evidence_request, format_credible_sources_for_prompt

PERSONA_PROMPTS = {
    "Public Health Official (Harm Reduction)": (
        "You are a senior public health official who supports harm reduction, health equity, "
        "person-first language, and access to care. You focus on reducing preventable harms, "
        "dignity, and practical policy tradeoffs. When asked for evidence, use only approved sources "
        "from the prompt and do not invent specifics."
    ),
    "Public Health Official (Abstinence / Zero-Tolerance)": (
        "You are a senior public health official who believes prevention is the strongest public-health strategy. "
        "You emphasize abstinence-based messaging, clear standards, and social norms framing. "
        "When asked for evidence, use only approved sources from the prompt and do not invent specifics."
    ),
    "State Legislator (Harm Reduction)": (
        "You are a two-term state legislator who supports harm reduction through a pragmatic, fiscally responsible lens. "
        "You argue with budget, implementation, and constituent safety framing. "
        "When asked for evidence, use only approved sources from the prompt and do not invent specifics."
    ),
    "State Legislator (Abstinence / Zero-Tolerance)": (
        "You are a state legislator with a public-safety and deterrence orientation, grounded in local control, parental rights, and rule-of-law framing. "
        "You emphasize enforcement feasibility, community norms, and implementation accountability. "
        "When asked for evidence, use only approved sources from the prompt and do not invent specifics."
    ),
}

PERSONA_KEY_MAP = {
    "Public Health Official (Harm Reduction)": "Public Health Official (Harm Reduction)",
    "Public Health Official (Abstinence / Zero-Tolerance)": "Public Health Official (Abstinence / Zero-Tolerance)",
    "State Legislator (Harm Reduction)": "State Legislator (Harm Reduction)",
    "State Legislator (Abstinence / Zero-Tolerance)": "State Legislator (Abstinence / Zero-Tolerance)",
}

ROLE_PROMPTS = {
    "Public Health Official": (
        "You are a senior public health official experienced in community health and evidence-informed policy. "
        "Focus on equity, implementation, and keeping people alive and connected to care."
    ),
    "State Legislator": (
        "You are a state legislator experienced in committee negotiations, budget tradeoffs, and constituent politics. "
        "Frame policy in terms of feasibility, fiscal realism, legal authority, and coalition-building."
    ),
    "Law Enforcement": (
        "You are a law enforcement leader focused on community safety, enforcement practicality, and operational impact. "
        "Use clear language about public order, deterrence, and police-community tradeoffs."
    ),
    "Clinician": (
        "You are a clinician working directly with patients affected by substance use. "
        "Prioritize patient safety, clinical evidence, care pathways, and treatment access realities."
    ),
    "Community Advocate": (
        "You are a community advocate representing people, families, and neighborhoods affected by substance use policy. "
        "Center lived experience, stigma reduction, and practical neighborhood impact."
    ),
}

ORIENTATION_PROMPTS = {
    "Minimal Harm Reduction": (
        "Your position is minimal harm reduction. You support narrow, evidence-informed interventions that prioritize overdose prevention, infection risk reduction, and care connection. "
        "Avoid broader criminal-justice reforms and keep the focus on pragmatic safety nets."
    ),
    "Harm Reduction": (
        "Your position is harm reduction. You prioritize policies that reduce overdose deaths, prevent infectious disease, and connect people to treatment. "
        "Support pragmatic interventions like naloxone, syringe services, safer supply, and treatment access."
    ),
    "Abstinence / Zero-Tolerance": (
        "Your position is abstinence and zero-tolerance. You emphasize prevention, clear standards, and enforcement. "
        "Question whether harm reduction normalizes risky behavior and favor policies that discourage use while noting long-term health goals."
    ),
}

_GREETING_EXACT = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "whats up",
    "what's up",
    "good morning",
    "good afternoon",
    "good evening",
}

_META_PREFIXES = (
    "who are you",
    "what are you",
    "introduce yourself",
    "what is your role",
    "what's your role",
    "tell me about yourself",
    "what do you care about",
)

_SMALL_TALK_PREFIXES = (
    "how are you",
    "how are you doing",
    "how's it going",
    "hows it going",
    "how have you been",
    "nice to meet you",
    "good to meet you",
    "thanks",
    "thank you",
)

_TOPIC_KEYWORDS = (
    "policy",
    "program",
    "programs",
    "public health",
    "health",
    "harm reduction",
    "overdose",
    "naloxone",
    "syringe",
    "consumption site",
    "injection site",
    "school",
    "schools",
    "housing",
    "treatment",
    "mental health",
    "clinic",
    "hospital",
    "fund",
    "funding",
    "expand",
    "ban",
    "legal",
    "legalize",
    "mandate",
    "require",
    "regulate",
    "vaccin",
    "mask",
)

_POLICY_CUES = (
    "should ",
    "should governments",
    "require",
    "mandate",
    "ban",
    "allow",
    "fund",
    "expand",
    "restrict",
    "prohibit",
    "regulate",
    "legalize",
    "legalise",
)

_LEADING_FILLER_PATTERNS = (
    r"^(i think|i believe|i'd argue|i would argue|my view is|in my view|from my perspective)\s+",
    r"^(can we debate|let['’]s debate(?: this)?|debate(?: this)?|discuss|talk about|argue about)\s*[:,]?\s+",
    r"^(actually[,!\s]+)?(let['’]s switch topics|lets switch topics|let['’]s change topics|lets change topics|switch topics|switch topic|new topic|different topic)\s*[:,]?\s+",
    r"^(fair point|good point|point taken|that['’]s fair|that is fair|i see your point|you make a fair point)[!,.\s]+",
    r"^(great|okay|ok|alright|all right|cool|awesome|thanks|thank you|so)[!,.\s]+",
    r"^(today\s+)?(i want to ask you|i want to talk about|my question is|the question is)\s*[:,]?\s+",
)

_TOPIC_SWITCH_PHRASES = (
    "new topic",
    "switch topics",
    "switch topic",
    "change topics",
    "change the topic",
    "different topic",
    "let's switch topics",
    "lets switch topics",
    "let's change topics",
    "lets change topics",
    "instead, let's debate",
    "instead lets debate",
)

_STRUCTURED_ANALYSIS_CUES = (
    "2 points on each side",
    "two points on each side",
    "on each side",
    "both sides",
    "strongest concerns critics raise",
    "strongest concerns critics",
    "counterargument",
    "counterarguments",
    "critics raise",
    "change your mind",
    "what data would change your mind",
    "what would change your mind",
)


def summarize_transcript(transcript: list[dict]) -> str:
    """Return a compact summary of all but the last 6 turns (max ~1200 chars)."""
    if len(transcript) <= 6:
        return ""
    older = transcript[:-6]
    lines = []
    for t in older:
        role = "Student" if t["role"] == "user" else "AI"
        lines.append(f"{role}: {t['content'][:120]}")
    summary = " | ".join(lines)
    return summary[:1200]


def _transcript_fingerprint(transcript: list[dict]) -> str:
    """Short hash of the full transcript — used to bust caches on every new turn."""
    raw = "||".join(f"{t['role']}:{t['content']}" for t in transcript)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_leading_fillers(text: str) -> str:
    cleaned = _normalise_whitespace(text)
    while cleaned:
        updated = cleaned
        for pattern in _LEADING_FILLER_PATTERNS:
            updated = re.sub(pattern, "", updated, flags=re.IGNORECASE).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _has_topic_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _TOPIC_KEYWORDS)


def _has_policy_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _POLICY_CUES)


def is_explicit_topic_switch_turn(text: str) -> bool:
    lowered = _normalise_whitespace(text).lower()
    return any(phrase in lowered for phrase in _TOPIC_SWITCH_PHRASES)


def topic_turn_strength(text: str) -> int:
    """
    Score whether a user turn looks like a real debate topic.

    Returns:
        0 = not a topic / small talk
        1 = plausible but weak topic signal
        2 = strong policy topic signal
    """
    cleaned = _normalise_whitespace(text)
    if not cleaned:
        return 0

    lowered = cleaned.lower().strip(" .!?")
    if lowered in _GREETING_EXACT:
        return 0
    if any(lowered.startswith(prefix) for prefix in _META_PREFIXES):
        return 0

    stripped = _strip_leading_fillers(cleaned)
    lowered = stripped.lower().strip(" .!?")
    if not lowered:
        return 0

    has_topic_signal = _has_topic_signal(lowered)
    has_policy_cue = _has_policy_cue(lowered)
    words = re.findall(r"\b\w+\b", lowered)

    if any(lowered.startswith(prefix) for prefix in _SMALL_TALK_PREFIXES):
        if not has_topic_signal and not has_policy_cue:
            return 0

    if len(words) < 4 and not has_topic_signal and not has_policy_cue:
        return 0
    if has_policy_cue and has_topic_signal:
        return 2
    if has_policy_cue and len(words) >= 6:
        return 2
    if has_topic_signal and ("?" in stripped or len(words) >= 8):
        return 1
    return 0


def is_substantive_debate_turn(text: str) -> bool:
    """Return True when a user turn contains an actual topic or policy claim."""
    return topic_turn_strength(text) > 0


def _clean_inferred_topic(raw_text: str) -> Optional[str]:
    cleaned = _normalise_whitespace(raw_text)
    cleaned = re.sub(r"^topic\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip("`'\"-: ")
    if not cleaned or cleaned.upper() == "NONE":
        return None
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rsplit(" ", 1)[0] + "..."
    cleaned = cleaned.rstrip(".")
    return cleaned[:1].upper() + cleaned[1:] if cleaned else None


def is_valid_topic_label(text: str) -> bool:
    cleaned = _clean_inferred_topic(text)
    if not cleaned:
        return False

    lowered = cleaned.lower()
    words = re.findall(r"\b\w+\b", lowered)
    if re.search(r"\b[a-z]{1,2}$", lowered):
        return False
    if len(words) < 3 and not _has_topic_signal(lowered) and not _has_policy_cue(lowered):
        return False
    return True


def _heuristic_topic_from_message(message: str) -> Optional[str]:
    if not is_substantive_debate_turn(message):
        return None

    cleaned = _strip_leading_fillers(message)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        return None

    topic = sentences[0].rstrip(".!?")
    for sentence in sentences:
        candidate = _strip_leading_fillers(sentence).rstrip(".!?")
        if not candidate:
            continue
        lowered = candidate.lower()
        if _has_policy_cue(lowered) or (_has_topic_signal(lowered) and "?" in sentence):
            topic = candidate
            break

    for pattern in _LEADING_FILLER_PATTERNS:
        topic = re.sub(pattern, "", topic, flags=re.IGNORECASE)

    topic = topic.strip(" :-")
    if not topic:
        return None
    if len(topic) > 120:
        topic = topic[:117].rsplit(" ", 1)[0] + "..."
    return topic[:1].upper() + topic[1:] if topic else None


def _build_topic_inference_prompt(message: str) -> str:
    return (
        "[topic-extractor]\n"
        "Extract the public-health debate topic from the student's message.\n"
        "Return one concise topic line only, between 6 and 18 words, with no quotation marks.\n"
        "If the message does not identify a concrete policy topic, return NONE.\n\n"
        f"STUDENT MESSAGE:\n{message}"
    )


def _wants_structured_analysis(text: str) -> bool:
    lowered = _normalise_whitespace(text).lower()
    if not lowered:
        return False
    return any(cue in lowered for cue in _STRUCTURED_ANALYSIS_CUES)


def infer_debate_topic(transcript: list[dict], llm_client=None) -> Optional[str]:
    """Infer and lock a debate topic from the most recent substantive user turn."""
    candidate = next(
        (
            turn["content"]
            for turn in reversed(transcript)
            if turn["role"] == "user" and is_substantive_debate_turn(turn["content"])
        ),
        "",
    )
    if not candidate:
        return None
    candidate = _strip_leading_fillers(candidate)
    heuristic_topic = _heuristic_topic_from_message(candidate)

    if heuristic_topic and topic_turn_strength(candidate) >= 2:
        return heuristic_topic

    client_info = {}
    if llm_client and hasattr(llm_client, "client_info"):
        try:
            client_info = llm_client.client_info()
        except Exception:
            client_info = {}

    if client_info.get("client") == "gemini":
        try:
            raw = llm_client.generate(
                _build_topic_inference_prompt(candidate),
                temperature=0.0,
                max_output_tokens=64,
            )
            inferred = _clean_inferred_topic(raw)
            if inferred and is_valid_topic_label(inferred):
                return inferred
        except Exception:
            pass

    return heuristic_topic


def _parse_persona_id(persona_id: str) -> tuple[str, str]:
    """Return (role, orientation) from a combined persona string."""
    if "(" in persona_id and persona_id.endswith(")"):
        role, orientation = persona_id.rsplit(" (", 1)
        orientation = orientation[:-1]
        return role.strip(), orientation.strip()
    return persona_id, ""


def build_debate_prompt(
    persona_id: str,
    debate_topic: str,
    transcript: list[dict],
    evidence_items: list[dict],
    retrieved_source_hits: list[dict] | None = None,
    include_source_catalog: bool = True,
) -> str:
    """
    Build a debate prompt for the given persona.

    Args:
        persona_id: display name or canonical key (e.g. "Public Health Official (Harm Reduction)")
        debate_topic: inferred topic/resolution for this debate session
        transcript: list of {"role": "user"|"assistant", "content": str}
                    NOTE: the latest user message must already be appended before calling this.
        evidence_items: list of {"title":str, "url":str, "bullets":list[str]}
        retrieved_source_hits: optional list of approved-domain page hits
    """
    key = PERSONA_KEY_MAP.get(persona_id, persona_id)
    role, orientation = _parse_persona_id(key)
    role_desc = ROLE_PROMPTS.get(role, "")
    orientation_desc = ORIENTATION_PROMPTS.get(orientation, "")
    persona_desc = PERSONA_PROMPTS.get(key)
    combined_persona_desc = "\n\n".join(
        part
        for part in (role_desc, orientation_desc, persona_desc)
        if part
    )
    if not combined_persona_desc:
        combined_persona_desc = persona_id
    topic_text = _normalise_whitespace(debate_topic)
    retrieved_source_hits = retrieved_source_hits or []

    # Evidence block
    evidence_block = ""
    for i, ev in enumerate(evidence_items, 1):
        bullets = "\n".join(f"  • {b}" for b in ev.get("bullets", []) if b.strip())
        evidence_block += f"\n[Evidence {i}] {ev['title']} ({ev.get('url', '')})\n{bullets}"

    # Transcript: summary of older turns + last 3 verbatim
    summary = summarize_transcript(transcript)
    recent = transcript[-3:]
    history = ""
    if summary:
        history += f"[Earlier context]: {summary}\n"
    for turn in recent:
        role = "Student" if turn["role"] == "user" else key
        history += f"\n{role}: {turn['content']}"

    # Extract the last user message explicitly for the instruction line
    last_user_msg = next(
        (t["content"] for t in reversed(transcript) if t["role"] == "user"), ""
    )
    evidence_request_mode = classify_evidence_request(last_user_msg)
    wants_sources = evidence_request_mode != "none"
    wants_structured_analysis = _wants_structured_analysis(last_user_msg)
    source_strict_mode = bool(topic_text)
    source_catalog_block = ""
    retrieved_hits_block = ""
    evidence_mode_block = ""
    source_strict_block = ""
    if retrieved_source_hits:
        hit_lines = []
        for index, hit in enumerate(retrieved_source_hits, 1):
            description = hit.get("description", "").strip()
            match_type = str(hit.get("match_type", "closest")).strip().lower()
            hit_lines.append(
                f"{index}. [{hit.get('source_name', 'Approved source')}] {hit.get('title', 'Untitled page')}\n"
                f"   Match: {match_type}\n"
                f"   URL: {hit.get('url', '')}"
            )
            if description:
                hit_lines.append(f"   Description: {description}")
        retrieved_hits_block = "RETRIEVED SOURCE HITS:\n" + "\n".join(hit_lines) + "\n\n"
    elif wants_sources:
        evidence_mode_block = f"EVIDENCE REQUEST MODE:\n{evidence_request_mode}\n\n"
        if not retrieved_source_hits:
            retrieved_hits_block = (
                "RETRIEVED SOURCE HITS:\n"
                "No exact approved-domain comparative study found for this phrasing.\n"
                "Closest relevant pages: none retrieved.\n\n"
            )
    if wants_sources:
        evidence_mode_block = f"EVIDENCE REQUEST MODE:\n{evidence_request_mode}\n\n"
    if source_strict_mode:
        if wants_sources and include_source_catalog:
            source_catalog_block = (
                "APPROVED SOURCE CATALOG:\n"
                f"{format_credible_sources_for_prompt(key, limit=3)}\n\n"
            )
        source_strict_block = (
            "SOURCE STRICT MODE: ON\n"
            "Use only EVIDENCE PROVIDED BY THE STUDENT or RETRIEVED SOURCE HITS from approved domains for empirical, causal, comparative, or numeric claims.\n"
            "Do not invent study titles, authors, dates, statistics, or URLs.\n"
            "If you cannot support a claim from approved sources, say 'I cannot support that claim with approved sources' and then offer policy reasoning.\n"
            "Speak clearly and humanly — this should sound like a thoughtful policy adviser, not a citation bot.\n\n"
        )

    # Fingerprint forces a unique prompt per transcript state — prevents cache collisions
    fingerprint = _transcript_fingerprint(transcript)

    response_constraint_block = ""
    if topic_text:
        topic_block = f"DEBATE TOPIC:\n{topic_text}\n\n"
        response_constraint_block = (
            "CONSTRAINT: Be concise. Keep responses under 3 sentences unless asked for an explanation. "
            "Focus on counter-arguments and rebuttals.\n\n"
        )
        claim_traceability_instruction = (
            "Use only student-provided evidence or retrieved approved-source hits for causal or empirical claims. "
            "Label inference clearly and do not invent study titles, authors, dates, statistics, data points, or URLs. "
            "If support is missing, say so and offer reasoned policy analysis."
        )
        if wants_sources:
            if evidence_request_mode == "hybrid_evidence":
                response_instruction = (
                    "The LAST STUDENT MESSAGE both challenges your reasoning and asks for evidence. "
                    "Answer that request directly with 1–2 short paragraphs or up to 4 compact bullets, under 180 words. "
                    "Use natural advisory language, not a checklist. "
                    "When possible, cite 1–2 page titles and exact URLs from RETRIEVED SOURCE HITS, and explain what each source helps evaluate. "
                    "Treat closest hits as related evidence, not direct proof. "
                    "Do not invent titles, URLs, study names, quotations, or statistics. "
                    + claim_traceability_instruction
                )
            else:
                response_instruction = (
                    "The LAST STUDENT MESSAGE is a request to support or source a prior claim. "
                    "Answer it directly with up to 3 compact bullets, under 140 words. "
                    "If RETRIEVED SOURCE HITS are available, list 1–3 concrete page titles with exact URLs and a short note on what they evaluate. "
                    "If only closest hits are available, explain they are related evidence, not direct proof. "
                    "If no RETRIEVED SOURCE HITS are available, say 'No exact approved-domain comparative study found for this phrasing.' "
                    + claim_traceability_instruction
                )
        elif wants_structured_analysis:
            response_instruction = (
                "The LAST STUDENT MESSAGE asks for a balanced, structured analysis. "
                "Answer the requested structure directly with short labeled points. "
                "Include 2 supporting points, 2 fair critic concerns, and one brief line on what data would change your mind. "
                "Use natural language and do not invent specifics. "
                + claim_traceability_instruction
            )
        else:
            response_instruction = (
                "Respond naturally to the LAST STUDENT MESSAGE, not to earlier turns. "
                "Use 3 sentences or fewer, under 120 words. Make a clear policy stance, mention one tradeoff, and include one fair concern. "
                "If you cannot cite a supporting URL, say 'based on policy and implementation tradeoffs' and avoid unsupported empirical claims. "
                + claim_traceability_instruction
            )
    else:
        topic_block = (
            "DEBATE TOPIC:\n"
            "Not identified yet. The student has not stated a concrete policy issue.\n\n"
        )
        response_instruction = (
            "If the LAST STUDENT MESSAGE does not identify a concrete public-health issue, "
            "ask one brief follow-up question to clarify the debate topic before making substantive claims. "
            "Do not invent a scenario or evidence. Stay in character and keep the reply under 40 words."
        )

    return (
        f"[turn-id:{fingerprint}]\n"
        f"ACTIVE PERSONA:\n{key}\n\n"
        f"You are playing the following role in a policy debate:\n{combined_persona_desc}\n\n"
        f"{topic_block}"
        f"{source_strict_block}"
        f"EVIDENCE PROVIDED BY THE STUDENT:\n{evidence_block or 'None yet.'}\n\n"
        f"{evidence_mode_block}"
        f"{retrieved_hits_block}"
        f"{source_catalog_block}"
        f"CONVERSATION SO FAR:\n{history}\n\n"
        f"LAST STUDENT MESSAGE:\n{last_user_msg}\n\n"
        f"{response_constraint_block}{response_instruction}"
    )
