# BUGFIX TASK: complete — score_turn_with_llm uses a transcript fingerprint in the
# scoring prompt so each call is unique and LLM caches cannot return stale scores.
"""
Persuasion scoring: Pydantic models, prompt builder, LLM-based scorer.

Reliability guarantees:
  1. Prompt explicitly forbids markdown — JSON only.
  2. On parse failure a "repair" prompt echoes the bad output and demands
     corrected JSON.
  3. If primary + repair fail, a compact retry prompt asks for a shorter JSON.
  4. Pydantic validates structure; scores are clamped 0-5 even if the model
     returns out-of-range values.
  5. List lengths are enforced post-validation:
       rewrite_suggestions : 2–3 items (padded / trimmed)
       reflection_prompts  : exactly 2 items (padded / trimmed)
  6. Local stigma-term check runs regardless of LLM output.
  7. A deterministic heuristic backup is returned if all LLM attempts fail.
"""

import hashlib
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Pydantic models ──────────────────────────────────────────────────────────

class PersuasionScores(BaseModel):
    ethos: int = Field(..., ge=0, le=5)
    logos: int = Field(..., ge=0, le=5)
    pathos: int = Field(..., ge=0, le=5)

    @model_validator(mode="before")
    @classmethod
    def clamp_scores(cls, values):
        """Clamp any out-of-range score to [0, 5] before Pydantic validates."""
        for key in ("ethos", "logos", "pathos"):
            if key in values:
                try:
                    values[key] = max(0, min(5, int(values[key])))
                except (TypeError, ValueError):
                    values[key] = 0
        return values


# Sentinel strings used to pad missing list items
_REWRITE_PAD = "Consider strengthening your argument with specific evidence or a direct citation."
_REFLECT_PAD = "What assumption in your argument might your opponent challenge most effectively?"

_MIN_REWRITES = 2
_MAX_REWRITES = 3
_EXACT_REFLECTIONS = 2


class ScoringOutput(BaseModel):
    scores: PersuasionScores
    rationale: dict  # {ethos: str, logos: str, pathos: str}
    rewrite_suggestions: list[str] = Field(default_factory=list)
    reflection_prompts: list[str] = Field(default_factory=list)
    ethics_flags: list[str] = Field(default_factory=list)
    suggested_rewrite: Optional[str] = None

    @field_validator("rewrite_suggestions")
    @classmethod
    def enforce_rewrites_length(cls, v: list[str]) -> list[str]:
        v = [s for s in v if s and s.strip()]           # drop blanks
        v = v[:_MAX_REWRITES]                            # trim to max 3
        while len(v) < _MIN_REWRITES:                    # pad to min 2
            v.append(_REWRITE_PAD)
        return v

    @field_validator("reflection_prompts")
    @classmethod
    def enforce_reflections_length(cls, v: list[str]) -> list[str]:
        v = [s for s in v if s and s.strip()]           # drop blanks
        v = v[:_EXACT_REFLECTIONS]                       # trim to max 2
        while len(v) < _EXACT_REFLECTIONS:               # pad to exactly 2
            v.append(_REFLECT_PAD)
        return v

    @field_validator("rationale", mode="before")
    @classmethod
    def ensure_rationale_keys(cls, v):
        if not isinstance(v, dict):
            v = {}
        for key in ("ethos", "logos", "pathos"):
            if key not in v or not str(v.get(key, "")).strip():
                v[key] = "No rationale provided."
        return v


_STIGMA_TERMS = [
    ("addict", "person who uses drugs"),
    ("junkie", "person struggling with addiction"),
    ("drug abuser", "person with a substance use disorder"),
    ("homeless people", "people experiencing homelessness"),
    ("the homeless", "people experiencing homelessness"),
    ("substance abuser", "person with a substance use disorder"),
]

# ── JSON schema embedded in prompt ───────────────────────────────────────────

_JSON_SCHEMA = """{
  "scores": {"ethos": <int 0-5>, "logos": <int 0-5>, "pathos": <int 0-5>},
  "rationale": {"ethos": "<max 12 words>", "logos": "<max 12 words>", "pathos": "<max 12 words>"},
  "rewrite_suggestions": ["<max 14 words>", "<max 14 words>"],
  "reflection_prompts": ["<max 12 words>", "<max 12 words>"],
  "ethics_flags": ["<flagged phrase or empty list>"],
  "suggested_rewrite": "<str or null>"
}"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_scoring_prompt(transcript: str, last_user_turn: str) -> str:
    # Fingerprint ensures a unique prompt per turn — prevents any upstream cache from
    # returning scores from a previous turn.
    fingerprint = hashlib.sha256(f"{transcript}|{last_user_turn}".encode()).hexdigest()[:12]
    return (
        f"[score-id:{fingerprint}]\n"
        "You are a debate coach evaluating a student's argument on a public health policy topic.\n\n"
        f"CONVERSATION CONTEXT:\n{transcript}\n\n"
        f"STUDENT'S LAST ARGUMENT (score THIS turn only — ignore prior turns):\n{last_user_turn}\n\n"
        "Score the student's last argument on Ethos (credibility), Logos (logic/evidence), "
        "and Pathos (emotional appeal), each 0–5.\n"
        "Flag any stigmatizing or non-person-first language and suggest a person-first rewrite if needed.\n"
        "Focus on persuasion strategy — do not express political opinions.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Output RAW JSON only. No markdown fences, no ```json, no explanation, no extra text.\n"
        "- Keep the entire response compact and under 220 tokens.\n"
        "- Keep every string brief. Prefer null for suggested_rewrite unless truly needed.\n"
        "- Your entire response must be a single valid JSON object matching this schema:\n"
        f"{_JSON_SCHEMA}"
    )


def _build_repair_prompt(original_prompt: str, bad_output: str) -> str:
    return (
        "The previous response was not valid JSON.\n"
        "Return only corrected minified JSON matching this schema.\n"
        "No markdown fences, no explanation, no extra text.\n\n"
        "Invalid output:\n"
        f"---\n{bad_output[:800]}\n---\n\n"
        f"Required schema:\n{_JSON_SCHEMA}"
    )


def _build_compact_retry_prompt(last_user_turn: str) -> str:
    return (
        "Return only minified JSON. No markdown, no explanation.\n"
        "Score THIS student argument on Ethos/Logos/Pathos (0-5):\n"
        f"{last_user_turn}\n\n"
        "Use this exact schema:\n"
        f"{_JSON_SCHEMA}"
    )


# ── JSON extraction & normalisation ──────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.
    Handles markdown fences, leading/trailing prose.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()

    # Try the whole text first (ideal case — pure JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(match.group())


def _normalise_flags(data: dict) -> dict:
    """Accept ethics_flags as list[str] or list[dict]."""
    flags = data.get("ethics_flags", [])
    if flags and isinstance(flags[0], dict):
        data["ethics_flags"] = [f.get("flagged", str(f)) for f in flags]
    return data


def parse_scoring_response(raw_text: str) -> ScoringOutput:
    data = _extract_json(raw_text)
    data = _normalise_flags(data)
    return ScoringOutput(**data)


# ── Local stigma check ────────────────────────────────────────────────────────

def _local_stigma_flags(text: str) -> tuple[list[str], Optional[str]]:
    found = [
        f'"{bad}" → consider "{good}"'
        for bad, good in _STIGMA_TERMS
        if bad in text.lower()
    ]
    rewrite = (
        "Consider replacing stigmatizing terms with person-first language throughout."
        if found else None
    )
    return found, rewrite


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _clamp_score(value: int) -> int:
    return max(0, min(5, int(value)))


def _heuristic_score_for_turn(last_user_turn: str) -> ScoringOutput:
    text = last_user_turn.strip()
    lowered = text.lower()
    words = re.findall(r"\b[a-z0-9'-]+\b", lowered)
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))

    source_terms = (
        "according to",
        "cdc",
        "nih",
        "nida",
        "who",
        "samhsa",
        "cochrane",
        "lancet",
        "kff",
        "ncsl",
        "report",
        "study",
        "evaluation",
    )
    evidence_terms = (
        "evidence",
        "data",
        "findings",
        "metrics",
        "outcome",
        "outcomes",
        "benchmark",
        "benchmarks",
    )
    logic_terms = (
        "because",
        "therefore",
        "if",
        "then",
        "however",
        "while",
        "so that",
        "which means",
        "tradeoff",
        "counterweight",
    )
    pathos_terms = (
        "community",
        "communities",
        "families",
        "children",
        "residents",
        "safety",
        "lives",
        "harm",
        "dignity",
        "equity",
        "fair",
        "compassion",
    )
    person_first_terms = (
        "people who",
        "people with",
        "persons who",
        "person-first",
    )
    has_number = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", lowered))

    ethos = 1
    if _contains_any(lowered, source_terms):
        ethos += 2
    elif _contains_any(lowered, evidence_terms):
        ethos += 1
    if _contains_any(lowered, person_first_terms):
        ethos += 1
    if _contains_any(lowered, ("i acknowledge", "fair point", "while", "however")):
        ethos += 1
    if len(words) >= 35:
        ethos += 1

    logos = 1
    if _contains_any(lowered, evidence_terms):
        logos += 1
    if has_number:
        logos += 1
    if _contains_any(lowered, logic_terms):
        logos += 1
    if sentence_count >= 2:
        logos += 1
    if _contains_any(lowered, ("rate", "rates", "cost", "costs", "metric", "metrics", "effect size")):
        logos += 1

    pathos = 1
    if _contains_any(lowered, pathos_terms):
        pathos += 1
    if _contains_any(lowered, ("urgent", "crisis", "protect", "prevent", "save lives")):
        pathos += 1
    if _contains_any(lowered, ("responsibility", "fairness", "justice", "equitable")):
        pathos += 1
    if len(words) >= 30 and _contains_any(lowered, ("community", "families", "residents")):
        pathos += 1

    ethos = _clamp_score(ethos)
    logos = _clamp_score(logos)
    pathos = _clamp_score(pathos)

    def _rationale(label: str, score: int) -> str:
        if label == "ethos":
            if score >= 4:
                return "Strong credibility signals and audience-aware framing."
            if score == 3:
                return "Moderate credibility; add clearer source attribution."
            return "Credibility is limited without explicit source grounding."
        if label == "logos":
            if score >= 4:
                return "Clear logic with evidence-oriented structure."
            if score == 3:
                return "Reasoning is present but needs stronger measurable support."
            return "Logic needs clearer evidence and explicit causal links."
        if score >= 4:
            return "Strong human-impact framing and persuasive empathy."
        if score == 3:
            return "Some emotional resonance; could better center affected groups."
        return "Limited emotional framing of concrete community impact."

    rewrite_suggestions: list[str] = []
    if logos <= 3:
        rewrite_suggestions.append("Add one concrete metric and cite its source in the same sentence.")
    if ethos <= 3:
        rewrite_suggestions.append("Name the organization or report behind your factual claim.")
    if pathos <= 3:
        rewrite_suggestions.append("Add one concrete resident-level impact to strengthen urgency.")
    if len(rewrite_suggestions) < 2:
        rewrite_suggestions.append("Acknowledge one tradeoff, then explain why your preferred policy still wins.")

    weakest = min((("ethos", ethos), ("logos", logos), ("pathos", pathos)), key=lambda x: x[1])[0]
    if weakest == "logos":
        reflection_prompts = [
            "Which one measurable outcome best tests your core claim?",
            "What opposing metric would most challenge your argument?",
        ]
    elif weakest == "ethos":
        reflection_prompts = [
            "What source would your opponent trust most for this claim?",
            "Where does your argument rely on assumption over citation?",
        ]
    else:
        reflection_prompts = [
            "Who is most affected, and how can you show that concretely?",
            "What value-based concern from critics should you acknowledge directly?",
        ]

    return ScoringOutput(
        scores=PersuasionScores(ethos=ethos, logos=logos, pathos=pathos),
        rationale={
            "ethos": _rationale("ethos", ethos),
            "logos": _rationale("logos", logos),
            "pathos": _rationale("pathos", pathos),
        },
        rewrite_suggestions=rewrite_suggestions,
        reflection_prompts=reflection_prompts,
        ethics_flags=[],
        suggested_rewrite=None,
    )


def _merge_local_stigma(result: ScoringOutput, last_user_turn: str) -> ScoringOutput:
    local_flags, local_rewrite = _local_stigma_flags(last_user_turn)
    if local_flags:
        result.ethics_flags = list(dict.fromkeys(result.ethics_flags + local_flags))
        result.suggested_rewrite = result.suggested_rewrite or local_rewrite
    return result


def _fallback_for_turn(last_user_turn: str) -> ScoringOutput:
    return _merge_local_stigma(_heuristic_score_for_turn(last_user_turn), last_user_turn)


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_turn_with_llm(
    transcript: list[dict],
    last_user_turn: str,
    llm_client,
) -> ScoringOutput:
    """
    Score the student's last turn using the LLM.

    Flow:
      1. Primary scoring prompt.
      2. Repair prompt on parse/shape failure.
      3. Compact retry prompt with smaller context.
      4. Deterministic heuristic backup if all attempts fail.
      5. Always merge local stigma-term flags.
    """
    transcript_str = "\n".join(
        f"{'Student' if t['role'] == 'user' else 'AI'}: {t['content']}"
        for t in transcript[-6:]
    )
    scoring_prompt = build_scoring_prompt(transcript_str, last_user_turn)

    raw = ""
    last_exc: Exception = RuntimeError("No attempts made")
    attempts: list[tuple[str, str, int]] = [
        ("primary", scoring_prompt, 800),
    ]

    for attempt_name, prompt, token_cap in attempts:
        try:
            raw = llm_client.generate(prompt, temperature=0.0, max_output_tokens=token_cap)
            result = parse_scoring_response(raw)
            return _merge_local_stigma(result, last_user_turn)
        except Exception as exc:
            last_exc = exc
            logger.info("Scoring %s attempt failed: %s | raw[:200]=%r", attempt_name, exc, raw[:200])

    repair_prompt = _build_repair_prompt(scoring_prompt, raw)
    try:
        raw = llm_client.generate(repair_prompt, temperature=0.0, max_output_tokens=600)
        result = parse_scoring_response(raw)
        return _merge_local_stigma(result, last_user_turn)
    except Exception as exc:
        last_exc = exc
        logger.info("Scoring repair attempt failed: %s | raw[:200]=%r", exc, raw[:200])

    compact_prompt = _build_compact_retry_prompt(last_user_turn)
    try:
        raw = llm_client.generate(compact_prompt, temperature=0.0, max_output_tokens=500)
        result = parse_scoring_response(raw)
        return _merge_local_stigma(result, last_user_turn)
    except Exception as exc:
        last_exc = exc
        logger.info("Scoring compact attempt failed: %s | raw[:200]=%r", exc, raw[:200])

    logger.warning("Scoring fell back to heuristic backup after 3 attempts: %s", last_exc)
    return _fallback_for_turn(last_user_turn)


# ── __main__ self-tests ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.llm_client import MockLLMClient  # noqa: E402

    # ── helpers ──────────────────────────────────────────────────────────────

    def _assert(condition: bool, msg: str) -> None:
        if not condition:
            raise AssertionError(f"FAIL: {msg}")
        print(f"  PASS: {msg}")

    def _run(label: str, transcript: list[dict], last_turn: str) -> ScoringOutput:
        print(f"\n{'─'*60}\nTest: {label}")
        client = MockLLMClient()
        result = score_turn_with_llm(transcript, last_turn, client)
        print(f"  scores : {result.scores}")
        print(f"  rewrites ({len(result.rewrite_suggestions)}): {result.rewrite_suggestions}")
        print(f"  reflections ({len(result.reflection_prompts)}): {result.reflection_prompts}")
        print(f"  flags   : {result.ethics_flags}")
        return result

    # ── Test 1: Normal evidence-heavy argument ────────────────────────────────
    t1_transcript = [
        {"role": "user", "content": "We need more funding for harm reduction programs."},
        {"role": "assistant", "content": "What evidence supports that claim?"},
    ]
    t1_turn = (
        "According to the CDC, syringe service programs reduce HIV transmission by 50%. "
        "The evidence is clear and we must act now for the sake of our community."
    )
    r1 = _run("Normal evidence-heavy argument", t1_transcript, t1_turn)
    _assert(isinstance(r1.scores.ethos, int), "ethos is int")
    _assert(0 <= r1.scores.ethos <= 5, f"ethos in [0,5]: {r1.scores.ethos}")
    _assert(0 <= r1.scores.logos <= 5, f"logos in [0,5]: {r1.scores.logos}")
    _assert(0 <= r1.scores.pathos <= 5, f"pathos in [0,5]: {r1.scores.pathos}")
    _assert(_MIN_REWRITES <= len(r1.rewrite_suggestions) <= _MAX_REWRITES,
            f"rewrite count in [2,3]: {len(r1.rewrite_suggestions)}")
    _assert(len(r1.reflection_prompts) == _EXACT_REFLECTIONS,
            f"reflection count == 2: {len(r1.reflection_prompts)}")

    # ── Test 2: Argument with stigma language ─────────────────────────────────
    t2_transcript = [{"role": "user", "content": "These addicts need help."}]
    t2_turn = "The homeless and junkies downtown need proper services, not punishment."
    r2 = _run("Stigma language argument", t2_transcript, t2_turn)
    _assert(len(r2.ethics_flags) > 0, "stigma flags detected")
    _assert(r2.suggested_rewrite is not None, "suggested_rewrite present when flags exist")
    _assert(_MIN_REWRITES <= len(r2.rewrite_suggestions) <= _MAX_REWRITES,
            f"rewrite count in [2,3]: {len(r2.rewrite_suggestions)}")
    _assert(len(r2.reflection_prompts) == _EXACT_REFLECTIONS,
            f"reflection count == 2: {len(r2.reflection_prompts)}")

    # ── Test 3: Out-of-range score clamping (simulate bad LLM JSON) ───────────
    print(f"\n{'─'*60}\nTest: Out-of-range score clamping")
    bad_json = json.dumps({
        "scores": {"ethos": 9, "logos": -2, "pathos": 3},
        "rationale": {"ethos": "great", "logos": "weak", "pathos": "ok"},
        "rewrite_suggestions": ["Fix A"],           # only 1 — should be padded to 2
        "reflection_prompts": ["Q1", "Q2", "Q3"],   # 3 — should be trimmed to 2
        "ethics_flags": [],
        "suggested_rewrite": None,
    })
    r3 = parse_scoring_response(bad_json)
    print(f"  scores (clamped): {r3.scores}")
    _assert(r3.scores.ethos == 5, f"ethos clamped to 5 (was 9): {r3.scores.ethos}")
    _assert(r3.scores.logos == 0, f"logos clamped to 0 (was -2): {r3.scores.logos}")
    _assert(r3.scores.pathos == 3, f"pathos unchanged at 3: {r3.scores.pathos}")
    _assert(_MIN_REWRITES <= len(r3.rewrite_suggestions) <= _MAX_REWRITES,
            f"rewrites padded to [2,3]: {len(r3.rewrite_suggestions)}")
    _assert(len(r3.reflection_prompts) == _EXACT_REFLECTIONS,
            f"reflections trimmed to 2: {len(r3.reflection_prompts)}")

    print(f"\n{'═'*60}")
    print("All tests passed ✅")
