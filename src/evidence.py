"""
Evidence schema, approved-source catalog, and prompt helpers.
"""

import re

VALID_SOURCES = {"PubTrawlr", "LegiScan"}

CREDIBLE_SOURCES = {
    1: {
        "name": "World Health Organization (WHO)",
        "url": "https://www.who.int",
        "urls": ["https://www.who.int"],
        "specialty": "Global health standards, international policy, disease surveillance",
        "best_for": "Both personas; Minimal Harm Reduction — global comparative data, international policy benchmarks",
    },
    2: {
        "name": "Centers for Disease Control and Prevention (CDC)",
        "url": "https://www.cdc.gov",
        "urls": ["https://www.cdc.gov"],
        "specialty": "U.S. epidemiology, disease prevention, behavioral health statistics",
        "best_for": "Both personas; Minimal Harm Reduction — U.S.-specific data on drug use, STIs, overdose rates",
    },
    3: {
        "name": "National Institutes of Health (NIH) / NIDA",
        "url": "https://www.nih.gov",
        "urls": ["https://www.nih.gov", "https://nida.nih.gov"],
        "specialty": "Biomedical research, addiction science, clinical trials",
        "best_for": "Harm Reduction; Minimal Harm Reduction — peer-reviewed addiction and treatment research",
    },
    4: {
        "name": "The Lancet",
        "url": "https://www.thelancet.com",
        "urls": ["https://www.thelancet.com"],
        "specialty": "High-impact peer-reviewed medical and public health research",
        "best_for": "Harm Reduction — frequently publishes harm reduction and drug policy studies",
    },
    5: {
        "name": "Substance Abuse and Mental Health Services Administration (SAMHSA)",
        "url": "https://www.samhsa.gov",
        "urls": ["https://www.samhsa.gov"],
        "specialty": "U.S. mental health and substance use policy, treatment data",
        "best_for": "Both personas; Minimal Harm Reduction — treatment outcomes, prevention program effectiveness",
    },
    6: {
        "name": "American Journal of Public Health (AJPH)",
        "url": "https://ajph.aphapublications.org",
        "urls": ["https://ajph.aphapublications.org"],
        "specialty": "Peer-reviewed public health policy and population health research",
        "best_for": "Both personas; Minimal Harm Reduction — equity research, policy evaluations, intervention studies",
    },
    7: {
        "name": "National Academy of Medicine (NAM)",
        "url": "https://nam.edu",
        "urls": ["https://nam.edu"],
        "specialty": "Consensus reports, evidence reviews, health policy recommendations",
        "best_for": "Abstinence/Zero-Tolerance — authoritative consensus reports carry policy weight",
    },
    8: {
        "name": "Cochrane Library",
        "url": "https://www.cochranelibrary.com",
        "urls": ["https://www.cochranelibrary.com"],
        "specialty": "Systematic reviews and meta-analyses — the gold standard for evidence",
        "best_for": "Both personas; Minimal Harm Reduction — strongest empirical evidence base for any intervention",
    },
    9: {
        "name": "Kaiser Family Foundation (KFF)",
        "url": "https://www.kff.org",
        "urls": ["https://www.kff.org"],
        "specialty": "U.S. health policy analysis, health equity, insurance and access data",
        "best_for": "Harm Reduction; Minimal Harm Reduction — health equity, access to care, Medicaid/coverage data",
    },
    10: {
        "name": "Office of National Drug Control Policy (ONDCP)",
        "url": "https://www.whitehouse.gov/ondcp",
        "urls": ["https://www.whitehouse.gov/ondcp"],
        "specialty": "Federal drug policy, enforcement strategy, prevention programs",
        "best_for": "Abstinence/Zero-Tolerance — federal prevention policy and enforcement data",
    },
    11: {
        "name": "National Conference of State Legislatures (NCSL)",
        "url": "https://www.ncsl.org",
        "urls": ["https://www.ncsl.org"],
        "specialty": "State-by-state policy comparisons, legislative tracking, bill summaries",
        "best_for": "Both personas; both legislators — neutral state policy tracking and legislative language",
    },
    12: {
        "name": "Pew Charitable Trusts / Stateline",
        "url": "https://www.pewtrusts.org",
        "urls": ["https://www.pewtrusts.org", "https://stateline.org"],
        "specialty": "State policy analysis, fiscal impact, criminal justice and health policy reporting",
        "best_for": "Harm Reduction; State Legislator (Harm Reduction) — fiscal and policy implementation framing",
    },
    13: {
        "name": "Vera Institute of Justice",
        "url": "https://www.vera.org",
        "urls": ["https://www.vera.org"],
        "specialty": "Criminal justice reform, incarceration costs, equity in enforcement",
        "best_for": "Harm Reduction; State Legislator (Harm Reduction) — incarceration cost and diversion outcomes",
    },
    14: {
        "name": "Bureau of Justice Statistics (BJS)",
        "url": "https://bjs.ojp.gov",
        "urls": ["https://bjs.ojp.gov"],
        "specialty": "Federal crime, enforcement, recidivism, and incarceration statistics",
        "best_for": "Both personas; both legislators — neutral justice-system baseline data",
    },
    15: {
        "name": "National Governors Association (NGA)",
        "url": "https://www.nga.org",
        "urls": ["https://www.nga.org"],
        "specialty": "State executive policy priorities, bipartisan opioid and health response briefs",
        "best_for": "Both personas; both legislators — cross-state implementation examples",
    },
    16: {
        "name": "Drug Enforcement Administration (DEA)",
        "url": "https://www.dea.gov",
        "urls": ["https://www.dea.gov"],
        "specialty": "Drug threat assessments, scheduling decisions, and federal enforcement data",
        "best_for": "Abstinence/Zero-Tolerance; State Legislator (Abstinence / Zero-Tolerance) — enforcement and supply-side evidence",
    },
    17: {
        "name": "National Drug Court Institute (NDCI)",
        "url": "https://www.ndci.org",
        "urls": ["https://www.ndci.org"],
        "specialty": "Drug court outcomes, diversion effectiveness, and cost comparisons",
        "best_for": "Both personas; both legislators — bipartisan diversion and recidivism evidence",
    },
    18: {
        "name": "Manhattan Institute",
        "url": "https://www.manhattan-institute.org",
        "urls": ["https://www.manhattan-institute.org"],
        "specialty": "Conservative policy research on criminal justice and urban policy",
        "best_for": "Abstinence/Zero-Tolerance; State Legislator (Abstinence / Zero-Tolerance) — right-leaning policy critiques",
    },
    19: {
        "name": "SAMHSA Dear Colleague Letter",
        "url": "https://www.samhsa.gov/sites/default/files/dear-colleague-letter-executive-order-ending-crime-disorder-americasstreets-07302025.pdf",
        "urls": [
            "https://www.samhsa.gov/sites/default/files/dear-colleague-letter-executive-order-ending-crime-disorder-americasstreets-07302025.pdf"
        ],
        "specialty": "Federal SAMHSA guidance on allowable and prohibited substance use prevention and treatment funding",
        "best_for": "Abstinence/Zero-Tolerance; State Legislator (Abstinence / Zero-Tolerance) — federal prevention policy and enforcement data",
    },
}

_SOURCE_ONLY_REQUEST_PATTERNS = (
    r"\b(?:can|could|would|will)\s+you\s+(?:show|give|share|provide|cite|reference|point\s+to|link)\b.*\b(?:source|sources|citation|citations|article|articles|study|studies|paper|papers|report|reports|proof|url|urls|link|links)\b",
    r"\b(?:back\s+that\s+up|support\s+that\s+claim|support\s+your\s+claim|source\s+for\s+that|sources\s+for\s+that|citation\s+for\s+that|citations\s+for\s+that|article\s+for\s+that|articles\s+for\s+that|where\s+is\s+that\s+from|where\s+did\s+you\s+get\s+that)\b",
    r"\b(?:cite|reference|source|sources|citation|citations|article|articles|study|studies|paper|papers|report|reports|proof|url|urls|link|links)\s+(?:that|this|it|please|for\s+that|your\s+claim|that\s+claim)\b",
    r"\b(?:what(?:'s| is)?\s+the\s+evidence|evidence|evidance)\s+(?:for|behind)\s+(?:that|this|it|your\s+claim|that\s+claim)\b",
    r"\bwhat\s+(?:evidence|data|study|studies|stats|statistics)\s+are\s+you\s+basing\s+(?:that|this|it)\s+on\b",
    r"\bwhat\s+are\s+you\s+relying\s+on\s+for\s+(?:that|this|it|your\s+claim|that\s+claim|that\s+concern|this\s+concern)\b",
    r"\bshow\s+me\s+where\s+(?:that|this|it)\s+comes\s+from\b",
    r"\bwhere\s+does\s+(?:that|this|it)\s+come\s+from\b",
    r"\bwhat(?:'s| is)\s+behind\s+(?:that|this|it|your\s+claim|your\s+caution)\b",
    r"\b(?:if\s+you\s+can\s+)?point\s+to\s+(?:one|a)\s+(?:credible|relevant|specific)\s+(?:example|source|study|paper|report|article)\b",
    r"^(?:sources|citations|refs|references|links|articles|studies|papers|reports|urls|evidence|proof|data)\??$",
)

_HYBRID_EVIDENCE_PATTERNS = (
    r"\bif\s+you\s+disagree,\s*cite\s+specific\s+(?:studies|papers|reports|public\s+health\s+reports)\b",
    r"\bif\s+you\s+think\s+that\s+evidence\s+does(?:\s+not|n't)\s+generalize\b",
    r"\bif\s+you\s+think\s+those\s+tradeoffs\s+outweigh\s+the\s+benefits\b",
    r"\bcite\s+specific\s+(?:studies|papers|reports|public\s+health\s+reports)\b",
    r"\bcite\s+(?:one|a)\s+specific\s+(?:study|paper|report|public\s+health\s+report|article)\b",
    r"\bcite\s+(?:one|a)\s+specific\s+(?:study|paper|report|public\s+health\s+report|article)\s*,?\s*(?:or\s+)?a\s+real[-\s]?world\s+case\b",
    r"\bwhat\s+evidence\s+are\s+you\s+relying\s+on\s+for\s+(?:that|this|it)\b",
    r"\bpoint\s+me\s+to\s+a\s+specific\s+(?:study|paper|report|article)(?:/[a-z-]+)?\b",
    r"\bpoint\s+me\s+to\s+a\s+specific\s+(?:study|paper|report|article)\s+or\s+a\s+real[-\s]?world\s+case\b",
    r"\bpoint\s+me\s+to\s+data\s+showing\b",
    r"\bpoint\s+me\s+to\s+(?:data|evidence)\s+showing\s+.*\b(?:increase|increases|worsen|worsens|harm|harms)\b",
    r"\bif\s+you\s+can(?:not|'t)\s+cite\s+credible\s+data\b",
    r"\bi(?:'m| am)\s+not\s+looking\s+for\s+empathy\b.*\b(?:proof|evidence)\b",
    r"\bi(?:'m| am)\s+looking\s+for\s+(?:proof|evidence)\b",
    r"\bwhat\s+evidence\s+do\s+you\s+have\s+that\b",
    r"\bi(?:'m| am)\s+not\s+looking\s+for\s+empathy\s*[-,]?\s*i(?:'m| am)\s+looking\s+for\s+(?:proof|evidence)\b",
    r"\bdo\s+you\s+have\s+anything\s+that\s+tracks\s+those\s+outcomes?\s+before\s+and\s+after\b",
    r"\bdo\s+you\s+have\s+anything\s+that\s+tracks\b.*\bbefore\s+and\s+after\b",
    r"\bare\s+we\s+just\s+hoping\b",
    r"\bpoint\s+to\s+a\s+real\s+example\b",
    r"\bground\s+(?:it|that|this|your\s+caution|the\s+tradeoff\s+claim)\s+in\s+(?:a\s+)?(?:real|specific|concrete)\s+(?:example|case|study|report|source)\b",
    r"\breal[-\s]?world\s+case\s+where\s+outcomes\s+did(?:\s+not|n't)\s+improve\b",
    r"\bstrongest,\s*most\s+direct\s+piece\s+of\s+evidence\b",
    r"\b(?:major\s+review|specific\s+city\s+evaluation)\b.*\bkey\s+findings\b",
    r"\bwhat\s+metric\s+would\s+you\s+use\s+to\s+judge\b",
    r"\btell\s+me\s+which\s+outcome\s+you\s+think\s+is\s+the\s+strongest\s+counterweight\b",
    r"\bwhat\s+(?:stud(?:y|ies)|program\s+evaluations?)\s+show\b",
    r"\bwhat\s+(?:data|evidence)\s+or\s+real[-\s]?world\s+example\s+are\s+you\s+relying\s+on\b",
    r"\bpoint\s+to\s+a\s+(?:study|program\s+evaluation)\s+where\b",
    r"\bwhat\s+outcomes?,?\s+what\s+effect\s+size,?\s+and\s+in\s+what\s+setting\b",
    r"\bwithout\s+evidence,?\s+['\"]?underperform['\"]?\s+is\s+just\s+a\s+label\b",
    r"\bgive\s+me\s+(?:one|a)\s+(?:specific|concrete)\s+example\s+with\s+numbers\b",
    r"\bwhere\s+has\s+needle\s+litter\s+.*\b(?:gone\s+down|decreased|dropped)\b",
    r"\bwhat\s+changes?\s+on\s+the\s+ground\b",
    r"\bwhere\s+has\s+.*\b(?:gone\s+down|decreased|dropped)\b.*\bby\s+how\s+much\b",
    r"\bif\s+it\s+does(?:\s+not|n't)\s+drop\b.*\bwhat\s+are\s+you\s+willing\s+to\s+(?:change|shut\s+down)\b",
    r"\bif\s+it\s+gets\s+worse\b.*\bwhat\s+are\s+you\s+willing\s+to\s+(?:change|shut\s+down)\b",
)

_EVIDENCE_REQUEST_PATTERNS = _SOURCE_ONLY_REQUEST_PATTERNS + _HYBRID_EVIDENCE_PATTERNS
_HYBRID_ARGUMENT_CUES = (
    "doesn't generalize",
    "does not generalize",
    "if you disagree",
    "if you think those tradeoffs outweigh the benefits",
    "counterargument",
    "counterarguments",
    "counterweight",
    "ground it in",
    "ground that in",
    "ground this in",
    "show me where that comes from",
    "where does that come from",
    "what are you relying on",
    "that concern",
    "this concern",
    "what's behind",
    "what is behind",
    "what metric would you use",
    "net positive",
    "point me to data showing",
    "real-world case",
    "real world case",
    "what changed after mitigation",
    "why do some communities",
    "meaningful community harms",
    "didn't improve",
    "did not improve",
    "underperform",
    "underperforms",
    "what data",
    "what studies",
    "program evaluation",
    "program evaluations",
    "what outcomes",
    "effect size",
    "in what setting",
    "with numbers",
    "tracks those outcomes",
    "before and after",
    "are we just hoping",
    "if you can't cite credible data",
    "if you cannot cite credible data",
    "looking for proof",
    "looking for evidence",
    "not looking for empathy",
    "what evidence do you have that",
    "what changes on the ground",
    "without evidence",
    "just a label",
    "back that up",
    "where has",
    "gone down",
    "by how much",
    "if it doesn't drop",
    "if it does not drop",
    "if it gets worse",
    "willing to change",
    "shut down",
)

_EVIDENCE_SIGNAL_TERMS = (
    "evidence",
    "data",
    "study",
    "studies",
    "program evaluation",
    "program evaluations",
    "evaluation",
    "evaluations",
    "report",
    "reports",
    "example",
    "examples",
    "numbers",
    "effect size",
    "benchmark",
    "benchmarks",
    "outcomes",
    "metric",
    "metrics",
    "before and after",
    "track",
    "tracks",
    "setting",
    "how much",
    "proof",
)

_CLAIM_CHALLENGE_CUES = (
    "relying on",
    "what are you relying on",
    "what evidence are you relying on",
    "what data are you relying on",
    "what are you relying on for that concern",
    "what are you relying on for this concern",
    "if you can't cite credible data",
    "if you cannot cite credible data",
    "looking for proof",
    "looking for evidence",
    "not looking for empathy",
    "what evidence do you have that",
    "tracks those outcomes",
    "before and after",
    "are we just hoping",
    "back that up",
    "support that claim",
    "without evidence",
    "just a label",
    "point to",
    "show me",
    "where does that come from",
    "where is that from",
    "underperform",
    "underperforms",
    "strongest argument against",
    "strongest objection",
    "what's your strongest objection",
    "what is your strongest objection",
    "strongest concern",
    "give me one concrete example",
    "give me one specific example",
    "what outcomes",
    "effect size",
    "in what setting",
    "what changes on the ground",
    "where has",
    "gone down",
    "by how much",
    "if it doesn't drop",
    "if it does not drop",
    "if it gets worse",
    "willing to change",
    "shut down",
)

_STRUCTURED_ANALYSIS_ONLY_CUES = (
    "2 points on each side",
    "two points on each side",
    "on each side",
    "both sides",
    "what data would change your mind",
)

_EVIDENCE_INTENT_SOURCE_TERMS = (
    "proof",
    "prove",
    "evidence",
    "basis",
    "citation",
    "citations",
    "source",
    "sources",
    "reference",
    "references",
    "study",
    "studies",
    "report",
    "reports",
    "paper",
    "papers",
    "url",
    "urls",
    "link",
    "links",
    "data",
    "stats",
    "statistics",
    "numbers",
)

_EVIDENCE_INTENT_TRUTH_CHALLENGE_TERMS = (
    "prove it",
    "prove that",
    "what's your basis",
    "what is your basis",
    "what are you basing",
    "what are you relying on",
    "what evidence are you relying on",
    "what data are you relying on",
    "what are you relying on for that",
    "what are you relying on for this",
    "show me",
    "point me to",
    "point to",
    "back that up",
    "support that claim",
    "not looking for empathy",
    "looking for proof",
    "looking for evidence",
    "without evidence",
    "just a label",
)

_EVIDENCE_INTENT_OUTCOME_TERMS = (
    "needle litter",
    "litter",
    "public disorder",
    "disorder",
    "overdose",
    "overdose rate",
    "overdose rates",
    "crime",
    "public use",
    "neighborhood safety",
    "neighbourhood safety",
    "hiv",
    "hcv",
    "hepatitis",
    "infection",
    "infectious",
)

_EVIDENCE_INTENT_QUANT_CUES = (
    "increase",
    "increases",
    "increased",
    "worsen",
    "worsens",
    "worsened",
    "decrease",
    "decreases",
    "decreased",
    "drop",
    "drops",
    "dropped",
    "rate",
    "rates",
    "outcome",
    "outcomes",
    "effect size",
    "metric",
    "metrics",
    "before and after",
    "by how much",
    "more",
    "less",
    "worse",
    "better",
)

_HARM_REDUCTION_PERSONA = "Public Health Official (Harm Reduction)"
_ABSTINENCE_PERSONA = "Public Health Official (Abstinence / Zero-Tolerance)"
_LEGISLATOR_HARM_REDUCTION_PERSONA = "State Legislator (Harm Reduction)"
_LEGISLATOR_ABSTINENCE_PERSONA = "State Legislator (Abstinence / Zero-Tolerance)"


def evidence_intent(text: str) -> str:
    """
    Lightweight intent detector to backstop regex-only evidence routing.

    Returns:
        "source_only" | "hybrid_evidence" | "none"
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    cleaned = (
        cleaned.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    if not cleaned:
        return "none"

    asks_question = "?" in cleaned
    has_proof_verb = bool(re.search(r"\b(?:prove|show|cite|reference|support|back)\b", cleaned))
    has_imperative_source_ask = bool(
        re.search(r"\b(?:cite|source|reference|link|url|show|point)\b", cleaned)
    )
    has_source_term = any(term in cleaned for term in _EVIDENCE_INTENT_SOURCE_TERMS)
    has_truth_challenge = any(term in cleaned for term in _EVIDENCE_INTENT_TRUTH_CHALLENGE_TERMS)
    has_outcome_term = any(term in cleaned for term in _EVIDENCE_INTENT_OUTCOME_TERMS)
    has_quant_cue = any(term in cleaned for term in _EVIDENCE_INTENT_QUANT_CUES) or bool(
        re.search(r"\b\d+(?:\.\d+)?%?\b", cleaned)
    )
    structured_only = any(cue in cleaned for cue in _STRUCTURED_ANALYSIS_ONLY_CUES)

    direct_source_tokens = ("citation", "citations", "source", "sources", "url", "urls", "link", "links")
    if has_source_term and not has_truth_challenge and not has_outcome_term and not structured_only:
        if any(token in cleaned for token in direct_source_tokens):
            return "source_only"

    if has_truth_challenge and (has_source_term or has_outcome_term or asks_question or has_proof_verb):
        return "hybrid_evidence"

    if has_source_term and (asks_question or has_imperative_source_ask):
        return "hybrid_evidence"

    # Outcome/quant questions default to evidence mode even without explicit
    # citation phrasing (e.g., "will needle litter increase?").
    if has_outcome_term and has_quant_cue and (asks_question or has_imperative_source_ask):
        return "hybrid_evidence"

    return "none"


def classify_evidence_request(text: str) -> str:
    """Classify a source/evidence ask as source_only, hybrid_evidence, or none."""
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    cleaned = (
        cleaned.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    if not cleaned:
        return "none"

    direct_match = any(re.search(pattern, cleaned) for pattern in _SOURCE_ONLY_REQUEST_PATTERNS)
    hybrid_match = any(re.search(pattern, cleaned) for pattern in _HYBRID_EVIDENCE_PATTERNS)

    if hybrid_match:
        return "hybrid_evidence"

    if direct_match:
        if (
            any(cue in cleaned for cue in _HYBRID_ARGUMENT_CUES)
            or any(cue in cleaned for cue in _CLAIM_CHALLENGE_CUES)
        ):
            return "hybrid_evidence"
        return "source_only"

    # Keep broad structured prompts out of citation mode unless they include
    # explicit source-seeking language.
    if any(cue in cleaned for cue in _STRUCTURED_ANALYSIS_ONLY_CUES):
        explicit_source_terms = (
            "cite",
            "source",
            "sources",
            "citation",
            "citations",
            "study",
            "studies",
            "report",
            "reports",
            "article",
            "articles",
            "paper",
            "papers",
            "url",
            "urls",
            "link",
            "links",
            "example",
            "examples",
            "program evaluation",
            "program evaluations",
            "evaluation",
            "evaluations",
        )
        if not any(term in cleaned for term in explicit_source_terms):
            return "none"

    intent_mode = evidence_intent(cleaned)
    if intent_mode != "none":
        return intent_mode

    has_evidence_signal = any(term in cleaned for term in _EVIDENCE_SIGNAL_TERMS)
    has_challenge_cue = any(cue in cleaned for cue in _CLAIM_CHALLENGE_CUES)
    if has_evidence_signal and has_challenge_cue:
        return "hybrid_evidence"

    # Lightweight direct-source fallback for shorter asks that mention source
    # artifacts but miss regex patterns.
    source_tokens = ("source", "sources", "citation", "citations", "url", "urls", "link", "links")
    if has_evidence_signal and any(token in cleaned for token in source_tokens):
        return "source_only"

    return "none"


def is_evidence_request(text: str) -> bool:
    return classify_evidence_request(text) != "none"

def source_url_list(source: dict) -> list[str]:
    urls = source.get("urls") or []
    if urls:
        return [str(url).strip() for url in urls if str(url).strip()]
    primary = str(source.get("url", "")).strip()
    return [primary] if primary else []


def format_source_urls(source: dict) -> str:
    return " / ".join(source_url_list(source))


def relevant_credible_sources(persona_id: str) -> list[dict]:
    """Return approved sources, ranked for the active persona."""
    role = ""
    orientation = ""
    if " (" in persona_id and persona_id.endswith(")"):
        role, orientation = persona_id.rsplit(" (", 1)
        orientation = orientation[:-1]
        role = role.strip().lower()
        orientation = orientation.strip().lower()

    is_legislator_persona = role == "state legislator"
    is_clinician_persona = role == "clinician"
    is_community_persona = role == "community advocate"
    is_minimal_harm = "minimal harm reduction" in orientation

    if "harm reduction" in orientation or "minimal harm reduction" in orientation:
        preferred_tag = "harm reduction"
    elif "abstinence" in orientation:
        preferred_tag = "abstinence/zero-tolerance"
    else:
        preferred_tag = ""

    ranked = []
    for source_id, source in CREDIBLE_SOURCES.items():
        best_for = source["best_for"].lower()
        specialty = source["specialty"].lower()
        rank = 0
        if preferred_tag and preferred_tag in best_for:
            rank += 2
        if is_minimal_harm and "minimal harm reduction" in best_for:
            rank += 3

        if is_legislator_persona:
            if "legislator" in best_for:
                rank += 2
            elif "both personas" in best_for:
                rank += 1
        elif is_clinician_persona:
            if any(term in specialty for term in ("clinical", "medical", "peer-reviewed", "evidence base", "addiction", "public health")):
                rank += 2
            elif "both personas" in best_for:
                rank += 1
        elif is_community_persona:
            if any(term in specialty for term in ("equity", "community", "policy", "state policy", "health equity")):
                rank += 2
            elif "both personas" in best_for:
                rank += 1
        else:
            if "both personas" in best_for:
                rank += 1

        if is_minimal_harm and any(term in specialty for term in ("overdose", "prevent", "treatment", "access", "implementation", "equity", "policy")):
            rank += 1

        ranked.append((rank, source_id, source))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [source for _, _, source in ranked]


def format_credible_sources_for_prompt(persona_id: str, limit: int | None = None) -> str:
    """Format the approved source catalog for prompt injection."""
    sources = relevant_credible_sources(persona_id)
    if limit is not None:
        sources = sources[:limit]

    lines = []
    for index, source in enumerate(sources, 1):
        lines.append(
            f"{index}. {source['name']} — {format_source_urls(source)} | "
            f"Specialty: {source['specialty']} | Best for: {source['best_for']}"
        )
    return "\n".join(lines)


def recommended_source_cards(persona_id: str, limit: int = 3) -> list[dict]:
    """Small persona-aware subset for deterministic mock replies."""
    return relevant_credible_sources(persona_id)[:limit]


def normalize_evidence_item(raw: dict) -> dict:
    """
    Normalize a raw evidence dict into the standard schema.
    Missing fields are filled with sensible defaults.
    """
    source = raw.get("source", "")
    if source not in VALID_SOURCES:
        source = "PubTrawlr"  # default

    key_points = raw.get("key_points") or raw.get("bullets") or []
    # Cap to 4 points, strip blanks
    key_points = [p.strip() for p in key_points if str(p).strip()][:4]

    date_raw = raw.get("date", "")
    date_str = str(date_raw).strip() if date_raw else ""

    return {
        "source": source,
        "title": str(raw.get("title", "Untitled")).strip(),
        "url": str(raw.get("url", "")).strip(),
        "date": date_str,
        "key_points": key_points,
    }


def format_evidence_for_prompt(evidence_items: list[dict]) -> str:
    """
    Format a list of evidence items as compact numbered bullets for an LLM prompt.
    Returns an empty string if the list is empty.
    """
    if not evidence_items:
        return ""

    lines = []
    for i, item in enumerate(evidence_items, 1):
        ev = normalize_evidence_item(item)
        date_part = f" ({ev['date']})" if ev["date"] else ""
        header = f"{i}. [{ev['source']}] {ev['title']}{date_part}"
        if ev["url"]:
            header += f" — {ev['url']}"
        lines.append(header)
        for point in ev["key_points"]:
            lines.append(f"   • {point}")

    return "\n".join(lines)


# ── Self-test when run directly ───────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        {
            "source": "PubTrawlr",
            "title": "CDC SSP Evaluation 2023",
            "url": "https://cdc.gov/ssp-eval-2023",
            "date": "2023-09-15",
            "key_points": [
                "35% reduction in needle-sharing among participants",
                "No statistically significant increase in local drug use",
            ],
        },
        {
            "source": "LegiScan",
            "title": "HB 4102 – Naloxone Access Act",
            "url": "https://legiscan.com/hb4102",
            "date": "",
            "bullets": [  # test alias field
                "Mandates naloxone availability in public high schools",
                "Allocates $2M in state funding for vending machines",
                "Requires staff training within 6 months of enactment",
            ],
        },
        {
            # minimal / missing fields
            "title": "Insite Vancouver Long-Term Study",
            "key_points": ["15-year operation without a single on-site overdose death"],
        },
    ]

    print("=== normalize_evidence_item ===")
    for s in samples:
        print(normalize_evidence_item(s))
        print()

    print("=== format_evidence_for_prompt ===")
    print(format_evidence_for_prompt(samples))
    print()
    print("=== empty list ===")
    print(repr(format_evidence_for_prompt([])))
