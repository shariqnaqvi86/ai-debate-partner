
"""
LLM client: full Gemini implementation + MockLLMClient for offline testing.

Usage:
    from src.llm_client import LLMClient, MockLLMClient

    # Real client (reads GEMINI_API_KEY from env)
    client = LLMClient()
    reply = client.generate("Your prompt here")

    # Offline / test client
    client = MockLLMClient()
"""

import hashlib
import json
import logging
import os
import re
import time
import datetime
import collections
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from src.evidence import format_source_urls, is_evidence_request, recommended_source_cards

# ── Log file lives in data/ relative to this file's parent ──────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LLM_LOG_FILE = os.path.join(_DATA_DIR, "llm_calls.jsonl")
_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

logger = logging.getLogger(__name__)

# ── Mock responses ───────────────────────────────────────────────────────────

_MOCK_GREETING_REPLY = (
    "Welcome to the debate. I'm here to challenge your arguments on this policy topic. "
    "Please share your opening position and I'll respond in character."
)

_MOCK_SHORT_REPLY = (
    "Which public-health policy issue do you want to debate? "
    "Share your claim and the outcome you want so I can respond in character."
)

_MOCK_HARM_REDUCTION_REPLIES = [
    "If our goal is to reduce preventable harm, then access, trust, and person-first services have to come first. "
    "How does your proposal reduce risk for the people most affected right now?",
    "Punitive or exclusionary policies often push people away from care rather than toward better outcomes. "
    "What evidence shows your approach improves health without increasing stigma or barriers to help?",
    "A harm-reduction lens asks whether the policy saves lives even before perfect behavior change happens. "
    "How does your proposal account for people who are not yet ready or able to comply fully?",
    "Upstream prevention matters, but so does keeping people alive long enough to benefit from it. "
    "Why is your approach better than expanding evidence-based support and safer access to services?",
]

_MOCK_ZERO_TOLERANCE_REPLIES = [
    "That framing skips key implementation tradeoffs. If we are serious about public-health outcomes, "
    "we need clear standards, strong compliance, and evidence that the policy discourages risky behavior rather than accommodating it. "
    "What metrics would you use to judge success?",
    "I hear the compassionate intent, but public-health policy also has to consider deterrence, norms, and accountability. "
    "How does your proposal avoid signaling that harmful behavior is acceptable?",
    "The strongest prevention strategy may be one that sets clear boundaries before risk escalates. "
    "What evidence shows your approach reduces initiation or long-term dependency, not just immediate harm?",
    "Every policy choice signals what the public sector is willing to normalize. "
    "Why is your approach better than stricter prevention, enforcement, or abstinence-focused interventions?",
]

_MOCK_SCORING_REPLY = json.dumps({
    "scores": {"ethos": 3, "logos": 2, "pathos": 3},
    "rationale": {
        "ethos": "Credibility is established but could be strengthened with explicit sourcing.",
        "logos": "The logical chain is present but lacks specific statistics.",
        "pathos": "Appeals to community impact are effective.",
    },
    "rewrite_suggestions": [
        "Try grounding your claim with a specific statistic: 'According to [source], …'",
        "Acknowledge the opposing view before pivoting: 'While some argue X, the evidence suggests …'",
    ],
    "reflection_prompts": [
        "Who is most affected by this policy, and whose voice is centered in your argument?",
        "What would a critic from the other side say is the weakest part of your last statement?",
    ],
    "ethics_flags": [],
    "suggested_rewrite": None,
})


# ── Shared helpers ───────────────────────────────────────────────────────────

def _log_call(
    model: str,
    prompt_len: int,
    success: bool,
    latency_ms: float = 0.0,
    cached: bool = False,
    error: str = "",
    prompt_hash: str = "",
) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "model": model,
        "prompt_hash": prompt_hash,
        "estimated_tokens": prompt_len // 4,
        "latency_ms": round(latency_ms, 1),
        "cached": cached,
        "success": success,
        "error": error,
    }
    with open(LLM_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if not success:
        logger.warning("LLM call failed — model=%s error=%s", model, error)


def _cache_key(prompt: str, temperature: float, max_tokens: int) -> str:
    raw = f"{prompt}|{temperature}|{max_tokens}"
    return hashlib.sha256(raw.encode()).hexdigest()


_FINISH_REASON_INT_MAP = {
    1: "stop",
    2: "max_tokens",
    3: "safety",
    4: "recitation",
    5: "other",
}


def _first_candidate(response) -> object | None:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0] if candidates else None


def _extract_response_text(response) -> str:
    try:
        text = str(response.text).strip()
        if text:
            return text
    except Exception:
        pass

    candidate = _first_candidate(response)
    content = getattr(candidate, "content", None) if candidate is not None else None
    parts = getattr(content, "parts", None) or []
    collected = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text:
            collected.append(str(part_text))
    return "".join(collected).strip()


def _extract_finish_reason(response) -> str:
    candidate = _first_candidate(response)
    reason = getattr(candidate, "finish_reason", None) if candidate is not None else None
    if reason is None:
        return ""
    if hasattr(reason, "name") and getattr(reason, "name"):
        return str(reason.name).strip().lower()
    if isinstance(reason, int):
        return _FINISH_REASON_INT_MAP.get(reason, str(reason))

    reason_text = str(reason).strip().lower()
    if "." in reason_text:
        reason_text = reason_text.split(".")[-1]
    return reason_text


def _text_looks_complete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{"):
        return stripped.endswith("}")
    if stripped.startswith("["):
        return stripped.endswith("]")
    if stripped.endswith(("...", "…", ",", ":", ";", "-", "—", "/", "(")):
        return False
    return bool(re.search(r'[.!?]["\')\]\}]*$', stripped))


def _response_is_acceptable(prompt: str, text: str, finish_reason: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if "[topic-extractor]" in prompt:
        return True

    if "Output RAW JSON only" in prompt:
        # For JSON-scoring prompts, accept any complete JSON object even when
        # finish_reason is max_tokens.
        return stripped.startswith("{") and stripped.endswith("}")

    reason = finish_reason.lower()
    if reason == "max_tokens":
        return _text_looks_complete(stripped)

    return _text_looks_complete(stripped)


def _can_salvage_partial_response(prompt: str, finish_reason: str) -> bool:
    reason = finish_reason.lower()
    if reason != "max_tokens":
        return False
    if "[topic-extractor]" in prompt:
        return False
    if "Output RAW JSON only" in prompt:
        return False
    return True


def _truncate_to_complete_sentences(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    sentence_endings = list(re.finditer(r'[.!?]["\')\]\}]*', stripped))
    if not sentence_endings:
        return ""

    candidate = stripped[:sentence_endings[-1].end()].strip()
    if len(candidate.split()) < 4:
        return ""
    return candidate if _text_looks_complete(candidate) else ""


def _extract_retrieved_hits(prompt: str) -> list[tuple[str, str, str, str]]:
    pattern = re.compile(
        r"\d+\.\s+\[(.*?)\]\s+(.*?)\n(?:\s+Match:\s*(exact|closest)\n)?\s+URL:\s*(https?://\S+)",
        flags=re.IGNORECASE,
    )
    hits = []
    for source, title, match_type, url in pattern.findall(prompt):
        normalized_match = (match_type or "exact").strip().lower()
        if normalized_match not in {"exact", "closest"}:
            normalized_match = "exact"
        hits.append((source.strip(), title.strip(), url.strip(), normalized_match))
    return hits


def _extract_evidence_mode(prompt: str) -> str:
    match = re.search(r"EVIDENCE REQUEST MODE:\n([^\n]+)", prompt)
    return match.group(1).strip() if match else "none"


def _normalize_url_token(token: str) -> str:
    return token.rstrip(").,;!?\"'")


def _extract_allowed_prompt_urls(prompt: str) -> set[str]:
    raw_urls = re.findall(r"https?://[^\s)]+", prompt)
    return {cleaned for cleaned in (_normalize_url_token(url) for url in raw_urls) if cleaned}


def _normalized_host(url: str) -> str:
    host = urlparse(url).netloc.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _best_prompt_url_match(candidate_url: str, allowed_urls: set[str]) -> str:
    if candidate_url in allowed_urls:
        return candidate_url

    candidate_host = _normalized_host(candidate_url)
    candidate_path = urlparse(candidate_url).path.rstrip("/")
    if not candidate_host:
        return ""

    matches: list[tuple[int, str]] = []
    for allowed_url in allowed_urls:
        allowed_host = _normalized_host(allowed_url)
        if not allowed_host:
            continue
        if not (allowed_host.startswith(candidate_host) or candidate_host.startswith(allowed_host)):
            continue

        score = 10
        allowed_path = urlparse(allowed_url).path.rstrip("/")
        if candidate_path and allowed_path.startswith(candidate_path):
            score += 4
        matches.append((score, allowed_url))

    matches.sort(key=lambda item: (-item[0], len(item[1])))
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) >= 2 and matches[0][0] > matches[1][0]:
        return matches[0][1]
    return ""


def _source_bucket(source_name: str, url: str) -> str:
    normalized_name = str(source_name or "").strip().lower()
    if normalized_name:
        return f"name:{normalized_name}"
    host = _normalized_host(url)
    return f"host:{host}"


def _select_diverse_hits(
    hits: list[tuple[str, str, str, str]],
    max_items: int = 2,
) -> list[tuple[str, str, str, str]]:
    if not hits or max_items <= 0:
        return []

    selected: list[tuple[str, str, str, str]] = []
    used_indexes: set[int] = set()
    seen_buckets: set[str] = set()

    for idx, hit in enumerate(hits):
        bucket = _source_bucket(hit[0], hit[2])
        if bucket in seen_buckets:
            continue
        selected.append(hit)
        used_indexes.add(idx)
        seen_buckets.add(bucket)
        if len(selected) >= max_items:
            return selected

    for idx, hit in enumerate(hits):
        if idx in used_indexes:
            continue
        selected.append(hit)
        if len(selected) >= max_items:
            break
    return selected


def _sanitize_response_urls(prompt: str, text: str) -> str:
    if "RETRIEVED SOURCE HITS:" not in prompt and "APPROVED SOURCE CATALOG:" not in prompt:
        return text

    allowed_urls = _extract_allowed_prompt_urls(prompt)
    if not allowed_urls:
        return text

    def replace_url(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        candidate_url = _normalize_url_token(raw_url)
        if candidate_url in allowed_urls:
            return raw_url.replace(candidate_url, candidate_url, 1)
        corrected = _best_prompt_url_match(candidate_url, allowed_urls)
        if corrected:
            return corrected
        return raw_url.replace(candidate_url, "", 1).strip()

    return re.sub(r"https?://[^\s)]+", replace_url, text)


_EMPIRICAL_CLAIM_CUES = (
    "evidence shows",
    "evidence suggests",
    "data show",
    "data suggests",
    "consistently shows",
    "overwhelmingly supports",
    "reduces",
    "reduce",
    "reduced",
    "decreases",
    "decrease",
    "increases",
    "increase",
    "worsens",
    "worsen",
    "improves",
    "improve",
    "underperform",
    "associated with",
    "linked to",
    "risk",
    "rates",
    "outcomes",
    "costs",
    "causes",
    "prevents",
    "does not",
    "do not",
)


def _response_has_allowed_source_url(prompt: str, text: str) -> bool:
    allowed_urls = _extract_allowed_prompt_urls(prompt)
    if not allowed_urls:
        return False
    mentioned_urls = {
        _normalize_url_token(url)
        for url in re.findall(r"https?://[^\s)]+", text or "")
    }
    if not mentioned_urls:
        return False
    for url in mentioned_urls:
        if url in allowed_urls:
            return True
        if _best_prompt_url_match(url, allowed_urls):
            return True
    return False


def _looks_empirical_claim(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    if any(cue in lowered for cue in _EMPIRICAL_CLAIM_CUES):
        return True
    # Numeric claims are often empirical/statistical.
    return bool(re.search(r"\b\d+(?:\.\d+)?%?\b", lowered))


def _extract_ordered_prompt_urls(prompt: str) -> list[str]:
    raw_urls = re.findall(r"https?://[^\s)]+", prompt or "")
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in raw_urls:
        cleaned = _normalize_url_token(raw)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _strip_unsourced_evidence_language(text: str) -> str:
    cleaned = str(text or "")
    patterns = (
        r"\bbased on general reasoning\s*\(not a sourced claim\)\s*,?\s*",
        r"\b(?:the\s+)?evidence\s+(?:consistently\s+)?(?:shows?|suggests?)\b",
        r"\b(?:the\s+)?data\s+(?:consistently\s+)?(?:shows?|suggests?)\b",
        r"\b(?:the\s+)?stud(?:y|ies)\s+(?:consistently\s+)?(?:show|shows|suggest|suggests)\b",
        r"\b(?:the\s+)?research\s+(?:consistently\s+)?(?:show|shows|suggest|suggests)\b",
        r"\b(?:it\s+is|it's)\s+proven\b",
        r"\b(?:this|that|it)\s+is\s+proven\b",
        r"\bproven\s+that\b",
        r"\bhas\s+been\s+proven\b",
        r"\b(?:it\s+)?proves?\b",
        r"\bdemonstrates?\b",
        r"\baccording\s+to\b",
        r"\bit(?:\s+is|'s)\s+clear\s+that\b",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:-")
    return cleaned


def _contains_unsourced_evidence_language(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    patterns = (
        r"\bevidence\s+(?:consistently\s+)?(?:shows?|suggests?)\b",
        r"\bdata\s+(?:consistently\s+)?(?:shows?|suggests?)\b",
        r"\bstud(?:y|ies)\s+(?:consistently\s+)?(?:show|shows|suggest|suggests)\b",
        r"\bresearch\s+(?:consistently\s+)?(?:show|shows|suggest|suggests)\b",
        r"\b(?:it\s+is|it's)\s+proven\b",
        r"\b(?:this|that|it)\s+is\s+proven\b",
        r"\bhas\s+been\s+proven\b",
        r"\bproven\s+that\b",
    )
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def _build_no_support_three_part_reply(
    base: str,
    prompt: str,
    closest_hits: list[tuple[str, str, str, str]],
) -> str:
    first = "I can't substantiate this exact claim with approved sources."

    if closest_hits:
        selected_hits = _select_diverse_hits(closest_hits, max_items=2)
        page_bits = [f"{title} ({url})" for _, title, url, _ in selected_hits if url]
        if page_bits:
            second = (
                "What I can support with citations is broader SSP-related context: "
                + "; ".join(page_bits)
                + " (context, not direct proof)."
            )
        else:
            second = (
                "What I can support with citations is broader SSP-related evidence, "
                "but no direct approved page supports this exact claim."
            )
    else:
        ordered_urls = _extract_ordered_prompt_urls(prompt)
        fallback_urls = ordered_urls[:2]
        if fallback_urls:
            second = (
                "What I can support with citations is broader SSP evidence from approved sources: "
                + "; ".join(fallback_urls)
                + "."
            )
        else:
            second = (
                "What I can support with citations is broader SSP evidence from approved sources, "
                "but no retrievable approved page was returned for this turn."
            )

    third = (
        "Interim guardrails (first 90 days): implement siting buffers where legally applicable, "
        "daily disposal and cleanup operations, complaint-response SLAs, and weekly public reporting "
        "of needle-litter complaints, 911 public-use calls, and cleanup response times with automatic review triggers if indicators worsen."
    )
    return f"{first} {second} {third}".strip()


def _to_reasoned_opinion(base: str) -> str:
    cleaned = re.sub(r"https?://[^\s)]+", "", base or "").strip()
    cleaned = _strip_unsourced_evidence_language(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    if not cleaned:
        cleaned = "the likely tradeoff is between speed of rollout and quality of safeguards"
    if cleaned and cleaned[0].isupper() and not (len(cleaned) >= 2 and cleaned[1].isupper()):
        cleaned = cleaned[0].lower() + cleaned[1:]
    return (
        "Based on general reasoning (not a sourced claim), "
        f"{cleaned}. "
        "This is an opinion/logic statement, not evidence."
    )


def _enforce_source_strict_claims(prompt: str, text: str) -> str:
    if "SOURCE STRICT MODE: ON" not in prompt:
        return text

    base = (text or "").strip()
    if not base:
        return text

    lowered = base.lower()
    if "i can only make empirical claims from approved listed sources" in lowered:
        return text

    evidence_mode = _extract_evidence_mode(prompt)
    retrieved_hits = _extract_retrieved_hits(prompt)
    exact_hits = [hit for hit in retrieved_hits if hit[3] == "exact"]
    closest_hits = [hit for hit in retrieved_hits if hit[3] == "closest"]
    has_allowed_url = _response_has_allowed_source_url(prompt, base)
    has_unsourced_evidence_language = _contains_unsourced_evidence_language(base)

    if evidence_mode != "none" and not exact_hits:
        return _build_no_support_three_part_reply(base, prompt, closest_hits)

    # Hard claim-honesty guardrail: do not allow evidence-signaling language
    # without an approved citation in the response.
    if has_unsourced_evidence_language and not has_allowed_url:
        if evidence_mode != "none":
            if exact_hits:
                selected_exact = _select_diverse_hits(exact_hits, max_items=2)
                page_bits = [f"{title} ({url})" for _, title, url, _ in selected_exact if url]
                if page_bits:
                    cleaned = _strip_unsourced_evidence_language(base)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
                    if cleaned and cleaned[0].isupper() and not (
                        len(cleaned) >= 2 and cleaned[1].isupper()
                    ):
                        cleaned = cleaned[0].lower() + cleaned[1:]
                    if not cleaned:
                        cleaned = "the strongest support I can provide is from these approved pages"
                    return (
                        f"Based on cited approved sources, {cleaned}. "
                        f"Source support: {'; '.join(page_bits)}."
                    ).strip()
            return _build_no_support_three_part_reply(base, prompt, closest_hits)
        return _to_reasoned_opinion(base)

    if not _looks_empirical_claim(base):
        return text
    if has_allowed_url:
        return text

    if retrieved_hits:
        selected_hits = _select_diverse_hits(exact_hits or retrieved_hits, max_items=2)
        page_bits = [f"{title} ({url})" for _, title, url, _ in selected_hits if url]
        if not page_bits:
            return text
        has_exact = any(match_type == "exact" for _, _, _, match_type in selected_hits) or bool(exact_hits)
        if base and not re.search(r'[.!?]["\')\]\}]*$', base):
            base += "."
        if has_exact:
            return f"{base} Source support: {'; '.join(page_bits)}.".strip()
        reasoned = _to_reasoned_opinion(base)
        return (
            f"{reasoned} Closest related approved pages (context, not direct proof): "
            f"{'; '.join(page_bits)}."
        ).strip()

    return _to_reasoned_opinion(base)


def _looks_thin_hybrid_reply(text: str) -> bool:
    lowered = (text or "").lower()
    word_count = len(re.findall(r"\b[a-z0-9'-]+\b", lowered))
    has_urls = bool(re.search(r"https?://", lowered))
    substantive_cues = (
        "because",
        "however",
        "tradeoff",
        "risk",
        "metric",
        "outcome",
        "strongest concern",
        "strongest objection",
        "strongest argument",
        "my objection",
        "my concern",
        "i would",
        "i'd",
        "policy",
        "implementation",
    )
    has_substance = any(cue in lowered for cue in substantive_cues)
    in_favor_only = ("in favor" in lowered or "support expanding" in lowered) and not has_substance
    citation_heavy = ("relevant pages:" in lowered or has_urls) and not has_substance
    if citation_heavy:
        return True
    return in_favor_only or (word_count < 45 and not has_substance)


def _ensure_hybrid_substance(prompt: str, text: str) -> str:
    if _extract_evidence_mode(prompt) != "hybrid_evidence":
        return text

    base = (text or "").strip()
    if not base:
        return text
    if "substantive answer:" in base.lower():
        return text
    if not _looks_thin_hybrid_reply(base):
        return text

    addition = (
        "Substantive answer: My strongest objection is not expansion itself, but under-designed rollout. "
        "If coverage, hours, and linkage pathways are weak, benefits are diluted. "
        "I would monitor early reach and care-linkage metrics to judge whether expansion is delivering full impact."
    )
    if base and not re.search(r'[.!?]["\')\]\}]*$', base):
        base += "."
    return f"{base} {addition}".strip()


def _ensure_retrieved_hit_urls(prompt: str, text: str) -> str:
    if "RETRIEVED SOURCE HITS:" not in prompt:
        return text
    if _extract_evidence_mode(prompt) == "none":
        return text

    retrieved_hits = _extract_retrieved_hits(prompt)
    if not retrieved_hits:
        return text

    mentioned_urls = {
        _normalize_url_token(url)
        for url in re.findall(r"https?://[^\s)]+", text)
    }
    base = text.strip()
    has_admission = "no exact approved-domain comparative study found" in base.lower()
    exact_hits = [hit for hit in retrieved_hits if hit[3] == "exact"]
    closest_hits = [hit for hit in retrieved_hits if hit[3] == "closest"]

    if exact_hits:
        if any(url in mentioned_urls for _, _, url, _ in exact_hits):
            return text
        selected_exact = _select_diverse_hits(exact_hits, max_items=2)
        page_bits = [f"{title} ({url})" for _, title, url, _ in selected_exact]
        if not page_bits:
            return text
        if base and not re.search(r'[.!?]["\')\]\}]*$', base):
            base += "."
        suffix = " Relevant pages: " + "; ".join(page_bits) + "."
        return f"{base}{suffix}".strip()

    if not closest_hits:
        return text

    admission = "No exact approved-domain comparative study found for this phrasing"
    if any(url in mentioned_urls for _, _, url, _ in closest_hits):
        if has_admission:
            return text
        if base and not re.search(r'[.!?]["\')\]\}]*$', base):
            base += "."
        return f"{base} {admission}.".strip()

    selected_closest = _select_diverse_hits(closest_hits, max_items=2)
    page_bits = [f"{title} ({url})" for _, title, url, _ in selected_closest]
    if not page_bits:
        return text
    closest_phrase = "closest relevant pages: " + "; ".join(page_bits) + "."
    if has_admission:
        if base and not re.search(r'[.!?]["\')\]\}]*$', base):
            base += "."
        return f"{base} {closest_phrase}".strip()
    if base and not re.search(r'[.!?]["\')\]\}]*$', base):
        base += "."
    return f"{base} {admission}; {closest_phrase}".strip()


def _ensure_no_hit_admission(prompt: str, text: str) -> str:
    if "RETRIEVED SOURCE HITS:" not in prompt:
        return text
    if _extract_evidence_mode(prompt) == "none":
        return text

    if _extract_retrieved_hits(prompt):
        return text

    if "no exact approved-domain comparative study found" in text.lower():
        return text

    base = text.strip()
    admission = "No exact approved-domain comparative study found for this phrasing."
    if not base:
        return admission
    if not re.search(r'[.!?]["\')\]\}]*$', base):
        base += "."
    return f"{admission} {base}".strip()


def _finalize_source_reply(prompt: str, text: str) -> str:
    sanitized = _sanitize_response_urls(prompt, text)
    substantive = _ensure_hybrid_substance(prompt, sanitized)
    with_urls = _ensure_retrieved_hit_urls(prompt, substantive)
    with_admission = _ensure_no_hit_admission(prompt, with_urls)
    return _enforce_source_strict_claims(prompt, with_admission)


def _extract_openai_output_text(payload: dict) -> str:
    """Best-effort extraction of response text from OpenAI Responses payloads."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text_obj = content.get("text")
            if isinstance(text_obj, str):
                if text_obj.strip():
                    chunks.append(text_obj.strip())
                continue
            if isinstance(text_obj, dict):
                value = text_obj.get("value")
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
                continue
            value = content.get("value")
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
    return "\n".join(chunks).strip()


# ── Mock client ──────────────────────────────────────────────────────────────

class MockLLMClient:
    """Deterministic offline client — useful for demos and unit tests."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self._debate_idx = 0

    def client_info(self) -> dict:
        return {"client": "mock", "model": self.model_name, "api_key_present": False}

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 800,
    ) -> str:
        # Scoring requests
        if "ethos" in prompt.lower() and "logos" in prompt.lower():
            reply = _MOCK_SCORING_REPLY
        else:
            # Extract the last user message — take only the first line after the label
            last_msg = ""
            if "LAST STUDENT MESSAGE:" in prompt:
                after = prompt.split("LAST STUDENT MESSAGE:")[-1]
                last_msg = after.strip().splitlines()[0].strip()
            persona_match = re.search(r"ACTIVE PERSONA:\n([^\n]+)", prompt)
            persona_key = (
                persona_match.group(1).strip()
                if persona_match
                else "Public Health Official (Harm Reduction)"
            )
            words = last_msg.lower().split()
            _greetings = {"hi", "hello", "hey", "greetings"}
            _pleasantries = {"how are you", "how are you doing", "how do you do", "good morning", "good afternoon", "good evening", "nice to meet you", "pleasure to meet you"}
            _identity_phrases = {"who are you", "what are you", "who r you", "introduce yourself", "your name", "tell me about yourself"}
            is_greeting = len(words) <= 2 and (not words or words[0] in _greetings)
            is_pleasantry = any(phrase in last_msg.lower() for phrase in _pleasantries)
            is_identity = any(phrase in last_msg.lower() for phrase in _identity_phrases)
            if is_greeting or is_pleasantry:
                reply = _MOCK_GREETING_REPLY
            elif is_identity:
                # Extract persona hint from prompt to give the right intro
                if "State Legislator" in persona_key and "Harm Reduction" in persona_key:
                    reply = (
                        "I'm a state legislator focused on harm reduction policy with a pragmatic fiscal lens. "
                        "I debate in terms of implementation, constituent outcomes, and evidence-backed tradeoffs."
                    )
                elif "State Legislator" in persona_key and "Abstinence" in persona_key:
                    reply = (
                        "I'm a state legislator focused on abstinence and zero-tolerance policy priorities. "
                        "I debate in terms of public safety, deterrence, and local-control implementation tradeoffs."
                    )
                elif "Clinician" in persona_key:
                    reply = (
                        "I'm a clinician who works directly with patients impacted by substance use. "
                        "I focus on treatment access, patient safety, and the real-world clinical consequences of policy."
                    )
                elif "Community Advocate" in persona_key:
                    reply = (
                        "I'm a community advocate representing people and families affected by substance use policy. "
                        "I focus on lived experience, stigma reduction, and the neighborhood-level impact of decisions."
                    )
                elif "Harm Reduction" in persona_key:
                    reply = (
                        "I'm a public health official focused on harm reduction, person-first language, and keeping people connected to care. "
                        "I'm here to debate your policy claim with you, so share the issue you want to argue."
                    )
                else:
                    reply = (
                        "I'm a public health official focused on abstinence-based prevention, strict standards, and zero-tolerance policy approaches. "
                        "I'm here to debate your policy claim with you, so share the issue you want to argue."
                    )
            elif is_evidence_request(last_msg):
                evidence_mode = _extract_evidence_mode(prompt)
                retrieved_hits = _extract_retrieved_hits(prompt)
                if evidence_mode == "hybrid_evidence" and len(retrieved_hits) >= 1:
                    reply = (
                        f"- {retrieved_hits[0][1]} ({retrieved_hits[0][2]}) is a relevant approved page. "
                        "It supports evaluating public-health and community-impact claims. "
                        "My short substantive answer is that visible disorder concerns are usually more about siting, operations, and disposal management than increased drug initiation."
                    )
                elif len(retrieved_hits) >= 2:
                    reply = (
                        f"A relevant page is {retrieved_hits[0][1]} from {retrieved_hits[0][0]} ({retrieved_hits[0][2]}). "
                        f"Another useful page is {retrieved_hits[1][1]} ({retrieved_hits[1][2]})."
                    )
                elif len(retrieved_hits) == 1:
                    reply = (
                        f"A relevant page is {retrieved_hits[0][1]} from {retrieved_hits[0][0]} "
                        f"({retrieved_hits[0][2]})."
                    )
                else:
                    cards = recommended_source_cards(persona_key, limit=3)
                    source_bits = [
                        f"{card['name']} ({format_source_urls(card)}) for {card['specialty'].lower()}"
                        for card in cards
                    ]
                    if evidence_mode == "hybrid_evidence":
                        reply = (
                            "I did not find an exact approved-domain page for that phrasing. "
                            f"I'd start with {source_bits[0]} and {source_bits[1]}, then use {source_bits[2]} for a broader review. "
                            "Substantively, I would treat litter or disorder concerns as implementation risks to monitor, not proof that SSPs increase drug use."
                        )
                    else:
                        reply = (
                            "I did not find an exact approved-domain page for that phrasing. "
                            f"I'd start with {source_bits[0]} and {source_bits[1]}. "
                            f"{source_bits[2]} is also useful if you want a broader policy or evidence review."
                        )
            elif len(words) <= 5:
                reply = _MOCK_SHORT_REPLY
            else:
                if "Harm Reduction" in persona_key:
                    reply = _MOCK_HARM_REDUCTION_REPLIES[self._debate_idx % len(_MOCK_HARM_REDUCTION_REPLIES)]
                else:
                    reply = _MOCK_ZERO_TOLERANCE_REPLIES[self._debate_idx % len(_MOCK_ZERO_TOLERANCE_REPLIES)]
                self._debate_idx += 1
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        _log_call(self.model_name, len(prompt), True, prompt_hash=ph)
        return reply


# ── Real Gemini client ───────────────────────────────────────────────────────

class LLMClient:
    """
    Google Gemini client with:
      - API key from GEMINI_API_KEY env var (or passed explicitly)
      - Configurable model (default: gemini-1.5-flash)
      - 3-attempt exponential-backoff retry on transient errors
      - Per-minute rate limiting
      - In-process prompt/response cache (avoids duplicate API calls)
      - Structured JSONL logging per call
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        calls_per_minute: int = 15,
        cache_enabled: bool = True,
    ):
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.calls_per_minute = calls_per_minute
        self.cache_enabled = cache_enabled

        self._call_times: collections.deque = collections.deque()
        self._cache: dict[str, str] = {}
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._openai_web_model = os.environ.get("OPENAI_WEB_SEARCH_MODEL", "gpt-5-mini")
        web_search_toggle = os.environ.get("OPENAI_WEB_SEARCH_ENABLED", "1").strip().lower()
        self._openai_web_search_enabled = bool(self._openai_api_key) and web_search_toggle not in {"0", "false", "no"}

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment variable "
                "or pass api_key= explicitly."
            )

        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=resolved_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._sdk_available = True
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

    def client_info(self) -> dict:
        return {
            "client": "gemini",
            "model": self.model_name,
            "api_key_present": True,
            "web_search_enabled": self._openai_web_search_enabled,
            "web_search_model": self._openai_web_model if self._openai_web_search_enabled else "",
        }

    def _should_use_web_search(self, prompt: str) -> bool:
        if not self._openai_web_search_enabled:
            return False
        if "Output RAW JSON only" in prompt:
            return False
        if "[topic-extractor]" in prompt:
            return False
        return "EVIDENCE REQUEST MODE:" in prompt

    def _generate_with_openai_web_search(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = {
            "model": self._openai_web_model,
            "input": prompt,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                }
            ],
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            _OPENAI_RESPONSES_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._openai_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DebateCoach/1.0",
            },
        )

        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            snippet = err_body[:180].replace("\n", " ")
            raise RuntimeError(f"OpenAI web search HTTP {exc.code}: {snippet}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenAI web search request failed: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI web search returned invalid JSON") from exc

        text = _extract_openai_output_text(parsed)
        if not text:
            raise RuntimeError("OpenAI web search returned no text output")
        return text

    # ── Rate limiter ─────────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        now = time.time()
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()
        if len(self._call_times) >= self.calls_per_minute:
            wait = 60 - (now - self._call_times[0]) + 0.1
            logger.debug("Rate limit reached — sleeping %.1fs", wait)
            time.sleep(max(0.0, wait))
        self._call_times.append(time.time())

    def _repair_truncated_response(self, prompt: str, partial_text: str) -> str:
        """Ask Gemini for a very short repaired rewrite of an incomplete prose reply."""
        if not partial_text.strip():
            return ""

        repair_prompt = (
            "[response-repair]\n"
            "The draft below was cut off mid-sentence.\n"
            "Rewrite it as a complete reply.\n"
            "Requirements:\n"
            "- Preserve the same stance and main point.\n"
            "- Do not add new facts, citations, statistics, or claims not already implied.\n"
            "- Return plain text only.\n"
            "- Keep it to 1-2 sentences, under 60 words.\n\n"
            f"ORIGINAL TASK:\n{prompt[:1200]}\n\n"
            f"INCOMPLETE DRAFT:\n{partial_text}"
        )

        self._rate_limit()
        response = self._model.generate_content(
            repair_prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 120,
            },
        )
        repaired = _extract_response_text(response)
        finish_reason = _extract_finish_reason(response)
        if _response_is_acceptable(repair_prompt, repaired, finish_reason):
            return repaired
        return ""

    def _continue_truncated_response(
        self,
        prompt: str,
        partial_text: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Ask Gemini to continue a reply that ended at max_tokens."""
        if not partial_text.strip():
            return ""

        continue_prompt = (
            "[response-continue]\n"
            "The previous reply was cut off at the end. Continue exactly where it left off, keeping it very short.\n"
            "Return plain text only.\n\n"
            f"ORIGINAL TASK:\n{prompt[:1000]}\n\n"
            f"INCOMPLETE DRAFT:\n{partial_text}"
        )

        self._rate_limit()
        response = self._model.generate_content(
            continue_prompt,
            generation_config={
                "temperature": max(0.1, float(temperature) - 0.1),
                "max_output_tokens": min(120, max_output_tokens),
            },
        )
        continued = _extract_response_text(response)
        finish_reason = _extract_finish_reason(response)
        if _response_is_acceptable(continue_prompt, continued, finish_reason):
            return continued.strip()
        return ""

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 800,
    ) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: The full prompt string.
            temperature: Sampling temperature (0–1). Lower = more deterministic.
            max_output_tokens: Maximum tokens in the response.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If all 3 retry attempts fail.
        """
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # Cache check
        if self.cache_enabled:
            key = _cache_key(prompt, temperature, max_output_tokens)
            if key in self._cache:
                _log_call(self.model_name, len(prompt), True, cached=True, prompt_hash=ph)
                return self._cache[key]

        if self._should_use_web_search(prompt):
            t0 = time.time()
            try:
                result = self._generate_with_openai_web_search(prompt, temperature, max_output_tokens)
                result = _finalize_source_reply(prompt, result)
                latency = (time.time() - t0) * 1000
                _log_call(
                    f"{self._openai_web_model}+web_search",
                    len(prompt),
                    True,
                    latency_ms=latency,
                    prompt_hash=ph,
                )
                if self.cache_enabled:
                    self._cache[key] = result  # type: ignore[possibly-undefined]
                return result
            except Exception as exc:
                logger.warning("OpenAI web search call failed (%s) — falling back to Gemini", exc)

        self._rate_limit()

        last_error = ""
        last_partial_text = ""
        last_finish_reason = ""
        for attempt in range(3):
            t0 = time.time()
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                result = _extract_response_text(response)
                finish_reason = _extract_finish_reason(response)
                if not _response_is_acceptable(prompt, result, finish_reason):
                    last_partial_text = result
                    last_finish_reason = finish_reason

                    salvaged = ""
                    if _can_salvage_partial_response(prompt, finish_reason):
                        salvaged = _truncate_to_complete_sentences(result)
                    if salvaged:
                        salvaged = _finalize_source_reply(prompt, salvaged)
                        latency = (time.time() - t0) * 1000
                        logger.warning(
                            "Gemini attempt %d/3 returned truncated output; salvaging complete sentence(s)",
                            attempt + 1,
                        )
                        _log_call(self.model_name, len(prompt), True, latency_ms=latency, prompt_hash=ph)
                        if self.cache_enabled:
                            self._cache[key] = salvaged  # type: ignore[possibly-undefined]
                        return salvaged

                    if finish_reason.lower() == "max_tokens":
                        continuation = self._continue_truncated_response(
                            prompt,
                            result,
                            temperature,
                            max_output_tokens,
                        )
                        if continuation:
                            combined = result.rstrip()
                            if not combined.endswith((".", "?", "!")):
                                combined += " " if combined else ""
                            combined = f"{combined}{continuation}".strip()
                            combined = _finalize_source_reply(prompt, combined)
                            latency = (time.time() - t0) * 1000
                            logger.warning(
                                "Gemini attempt %d/3 returned max_tokens; continued response successfully",
                                attempt + 1,
                            )
                            _log_call(self.model_name, len(prompt), True, latency_ms=latency, prompt_hash=ph)
                            if self.cache_enabled:
                                self._cache[key] = combined  # type: ignore[possibly-undefined]
                            return combined

                    last_error = (
                        "Incomplete Gemini response"
                        f" (finish_reason={finish_reason or 'unknown'}, text={result[:120]!r})"
                    )
                    backoff = 2 ** attempt
                    logger.warning(
                        "Gemini attempt %d/3 returned incomplete output (%s) — retrying in %ds",
                        attempt + 1, last_error, backoff,
                    )
                    time.sleep(backoff)
                    continue

                latency = (time.time() - t0) * 1000
                result = _finalize_source_reply(prompt, result)
                _log_call(self.model_name, len(prompt), True, latency_ms=latency, prompt_hash=ph)

                if self.cache_enabled:
                    self._cache[key] = result  # type: ignore[possibly-undefined]
                return result

            except Exception as exc:
                last_error = str(exc)
                backoff = 2 ** attempt
                logger.warning(
                    "Gemini attempt %d/3 failed (%s) — retrying in %ds",
                    attempt + 1, last_error, backoff,
                )
                time.sleep(backoff)

        if _can_salvage_partial_response(prompt, last_finish_reason):
            try:
                repaired = self._repair_truncated_response(prompt, last_partial_text)
                if repaired:
                    repaired = _finalize_source_reply(prompt, repaired)
                    logger.warning("Gemini reply required repair after repeated truncation")
                    _log_call(self.model_name, len(prompt), True, prompt_hash=ph)
                    if self.cache_enabled:
                        self._cache[key] = repaired  # type: ignore[possibly-undefined]
                    return repaired
            except Exception as exc:
                last_error = f"{last_error}; repair failed: {exc}"

        _log_call(self.model_name, len(prompt), False, error=last_error, prompt_hash=ph)
        raise RuntimeError(f"LLMClient failed after 3 retries: {last_error}")

    def clear_cache(self) -> None:
        """Clear the in-process response cache."""
        self._cache.clear()
