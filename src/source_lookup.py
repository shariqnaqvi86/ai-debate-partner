"""
Best-effort source lookup for evidence requests.

The lookup stays constrained to approved source domains. It discovers internal
URLs via robots.txt sitemap declarations first, then falls back to homepage
links if needed. The caller can pass the resulting page hits into the debate
prompt so the model cites concrete URLs instead of only naming an organization.
"""

from __future__ import annotations

from functools import lru_cache
import html
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from src.evidence import is_evidence_request, relevant_credible_sources, source_url_list


_USER_AGENT = "Mozilla/5.0 (compatible; DebateCoachSourceLookup/1.0)"
_REQUEST_TIMEOUT = 12
_MAX_BYTES = 2_000_000
_MAX_SITEMAP_DEPTH = 2
_MAX_CHILD_SITEMAPS = 12
_MAX_SOURCE_CANDIDATES = 18
_MAX_PAGE_FETCHES = 8
_MAX_QUERY_TERMS = 16
_GENERIC_SIGNAL_TERMS = {
    "community",
    "communities",
    "disease",
    "drug",
    "drugs",
    "encouraging",
    "encourages",
    "engagement",
    "harm",
    "impact",
    "public",
}
_PRIORITY_QUERY_TERMS = {
    "abstinence",
    "buprenorphine",
    "discarded",
    "disorder",
    "drug",
    "drugs",
    "employment",
    "exchange",
    "hcv",
    "hepatitis",
    "hiv",
    "immunization",
    "immunizations",
    "immunisation",
    "immunisations",
    "injection",
    "mandate",
    "mandates",
    "methadone",
    "naloxone",
    "needle",
    "needles",
    "opioid",
    "overdose",
    "school",
    "schools",
    "syringe",
    "travel",
    "vaccination",
    "vaccinations",
    "vaccine",
    "vaccines",
    "zero-tolerance",
}
_LOW_SIGNAL_QUERY_TERMS = {
    "consistent",
    "direct",
    "disorder",
    "findings",
    "just",
    "key",
    "most",
    "outcomes",
    "relying",
    "specific",
    "strongest",
}
_BACKSTOP_NOISE_TERMS = {
    "acceptance",
    "benchmark",
    "benchmarks",
    "concrete",
    "debate",
    "focusing",
    "guarantee",
    "heavily",
    "immediate",
    "least",
    "like",
    "linkage",
    "metric",
    "metrics",
    "mischaracterizing",
    "mischaracterizes",
    "participants",
    "preference",
    "primary",
    "reasoning",
    "referral",
    "referrals",
    "risk-reduction",
    "risks",
    "success",
}
_QUERY_ALIAS_GROUPS = (
    (
        "needle exchange",
        "needle exchanges",
        "injection drug use",
        "inject drugs",
        "needle litter",
        "people who inject drugs",
        "persons who inject drugs",
        "syringe service program",
        "syringe service programs",
        "syringe services program",
        "syringe services programs",
        "ssp",
        "ssps",
    ),
    (
        "vaccine",
        "vaccines",
        "vaccination",
        "vaccinations",
        "immunization",
        "immunizations",
        "immunisation",
        "immunisations",
    ),
    (
        "safe consumption site",
        "safe consumption sites",
        "supervised consumption site",
        "supervised consumption sites",
        "overdose prevention center",
        "overdose prevention centers",
    ),
    (
        "medication assisted treatment",
        "medication-assisted treatment",
        "medications for opioid use disorder",
        "mat",
        "moud",
        "buprenorphine",
        "methadone",
    ),
)
_STOPWORDS = {
    "and",
    "are",
    "about",
    "after",
    "against",
    "any",
    "appreciate",
    "argue",
    "asking",
    "because",
    "between",
    "but",
    "can",
    "challenge",
    "challenges",
    "claim",
    "claims",
    "city",
    "communities",
    "community",
    "connecting",
    "convince",
    "could",
    "care",
    "evidence",
    "evidence-based",
    "expand",
    "existing",
    "for",
    "from",
    "have",
    "health",
    "here",
    "i",
    "into",
    "it",
    "its",
    "impact",
    "major",
    "manage",
    "managing",
    "metrics",
    "more",
    "not",
    "okay",
    "organization",
    "organizations",
    "people",
    "perspective",
    "point",
    "preventing",
    "prioritize",
    "program",
    "programs",
    "public",
    "report",
    "reports",
    "raising",
    "reference",
    "relevant",
    "require",
    "required",
    "safeguards",
    "service",
    "services",
    "should",
    "source",
    "support",
    "that",
    "the",
    "their",
    "they",
    "this",
    "today",
    "tool",
    "think",
    "through",
    "use",
    "uses",
    "using",
    "vital",
    "what",
    "working",
    "which",
    "with",
    "you",
    "your",
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _domain_root(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _source_bucket(source_name: str, url: str) -> str:
    name = _normalize_space(source_name).lower()
    if name:
        return f"name:{name}"
    host = urlparse(url).netloc.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return f"host:{host}"


def _allowed_netlocs(source: dict) -> tuple[str, ...]:
    netlocs = []
    for raw_url in source_url_list(source):
        netloc = urlparse(raw_url).netloc.lower()
        if netloc:
            netlocs.append(netloc)
    return tuple(dict.fromkeys(netlocs))


def _is_allowed_url(candidate_url: str, allowed_netlocs: tuple[str, ...]) -> bool:
    netloc = urlparse(candidate_url).netloc.lower()
    if not netloc:
        return False
    return any(netloc == allowed or netloc.endswith(f".{allowed}") for allowed in allowed_netlocs)


def _tokenize_query(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9-]{2,}", (text or "").lower())
    counts: dict[str, int] = {}
    for word in words:
        if word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            0 if item[0] in _PRIORITY_QUERY_TERMS else 1,
            -item[1],
            -len(item[0]),
            item[0],
        ),
    )
    return [word for word, _ in ranked[:_MAX_QUERY_TERMS]]


def _append_unique_terms(target: list[str], candidates: list[str] | tuple[str, ...]) -> None:
    for candidate in candidates:
        term = _normalize_space(candidate).lower()
        if not term:
            continue
        if term in _STOPWORDS or term in _LOW_SIGNAL_QUERY_TERMS or term in _BACKSTOP_NOISE_TERMS:
            continue
        if len(term) < 3:
            continue
        if term not in target:
            target.append(term)


def _assistant_context(transcript: list[dict]) -> str:
    for turn in reversed(transcript[:-1] if transcript else transcript):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def _first_user_context(transcript: list[dict]) -> str:
    for turn in transcript:
        if turn.get("role") == "user":
            return turn.get("content", "")
    return ""


def _expand_alias_terms(text: str) -> tuple[str, ...]:
    lowered = _normalize_space(text).lower()
    expanded: list[str] = []
    for alias_group in _QUERY_ALIAS_GROUPS:
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in alias_group):
            expanded.extend(alias_group)
    return tuple(dict.fromkeys(expanded))


def _matching_alias_groups(text: str) -> tuple[tuple[str, ...], ...]:
    lowered = _normalize_space(text).lower()
    matches = []
    for alias_group in _QUERY_ALIAS_GROUPS:
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in alias_group):
            matches.append(alias_group)
    return tuple(matches)


def _text_matches_alias_group(text: str, alias_group: tuple[str, ...]) -> bool:
    normalized = _normalize_match_text(text)
    if not normalized:
        return False
    return any(_normalize_match_text(alias) in normalized for alias in alias_group)


def _matches_required_alias_groups(text: str, alias_groups: tuple[tuple[str, ...], ...]) -> bool:
    if not alias_groups:
        return True
    return any(_text_matches_alias_group(text, alias_group) for alias_group in alias_groups)


def _is_ssp_context(alias_groups: tuple[tuple[str, ...], ...]) -> bool:
    return any(
        "syringe service program" in alias_group or "needle exchange" in alias_group
        for alias_group in alias_groups
    )


def _query_context_parts(debate_topic: str, transcript: list[dict]) -> list[str]:
    parts = [
        debate_topic,
        _first_user_context(transcript),
        _assistant_context(transcript),
    ]
    latest_user = next((turn["content"] for turn in reversed(transcript) if turn.get("role") == "user"), "")
    parts.append(latest_user)
    prior_user = next(
        (
            turn["content"]
            for turn in reversed(transcript[:-1] if transcript else transcript)
            if turn.get("role") == "user"
        ),
        "",
    )
    parts.append(prior_user)
    return [_normalize_space(part) for part in parts if part]


def _build_query_terms(debate_topic: str, transcript: list[dict]) -> tuple[str, ...]:
    normalized_parts = _query_context_parts(debate_topic, transcript)
    combined = " ".join(normalized_parts)
    terms = list(_tokenize_query(combined))
    for alias_text in _expand_alias_terms(combined):
        for alias_term in _tokenize_query(alias_text):
            if alias_term not in terms:
                terms.append(alias_term)
    return tuple(terms[:_MAX_QUERY_TERMS])


def _topic_support_bundle(
    alias_groups: tuple[tuple[str, ...], ...],
    lowered_context: str,
) -> tuple[str, ...]:
    support_terms: list[str] = []

    for alias_group in alias_groups:
        if "syringe service program" in alias_group or "needle exchange" in alias_group:
            _append_unique_terms(
                support_terms,
                (
                    "syringe",
                    "needle",
                    "exchange",
                    "ssp",
                    "ssps",
                    "persons",
                    "inject",
                    "drugs",
                    "hiv",
                    "hepatitis",
                    "overdose",
                ),
            )
            if any(
                cue in lowered_context
                for cue in (
                    "access",
                    "coverage",
                    "equity",
                    "equitable",
                    "reach",
                    "barrier",
                    "barriers",
                    "hours",
                    "mobile",
                    "rural",
                    "zip code",
                    "zip codes",
                )
            ):
                _append_unique_terms(support_terms, ("access", "reach", "barriers", "coverage"))
            if any(
                cue in lowered_context
                for cue in (
                    "litter",
                    "needle litter",
                    "public use",
                    "public disorder",
                    "disorder",
                    "crime",
                    "loitering",
                    "schools",
                    "school",
                    "parks",
                    "public safety",
                    "911",
                    "complaints",
                    "cleanup",
                    "clean up",
                )
            ):
                _append_unique_terms(
                    support_terms,
                    (
                        "needle",
                        "litter",
                        "public safety",
                        "complaints",
                        "crime",
                        "schools",
                        "parks",
                        "community",
                    ),
                )
            if any(
                cue in lowered_context
                for cue in (
                    "referral",
                    "referrals",
                    "treatment",
                    "linkage",
                    "care",
                    "warm-handoff",
                    "warm handoff",
                )
            ):
                _append_unique_terms(support_terms, ("treatment", "care", "referral", "linkage"))
        elif "vaccine" in alias_group or "vaccination" in alias_group:
            _append_unique_terms(
                support_terms,
                (
                    "vaccine",
                    "vaccination",
                    "immunization",
                    "coverage",
                    "outbreak",
                    "school",
                    "employment",
                    "travel",
                ),
            )
        elif "safe consumption site" in alias_group or "overdose prevention center" in alias_group:
            _append_unique_terms(
                support_terms,
                (
                    "overdose",
                    "consumption",
                    "site",
                    "fatal",
                    "emergency",
                    "naloxone",
                    "supervised",
                ),
            )
        elif "medication assisted treatment" in alias_group or "moud" in alias_group:
            _append_unique_terms(
                support_terms,
                (
                    "treatment",
                    "buprenorphine",
                    "methadone",
                    "retention",
                    "opioid",
                    "overdose",
                ),
            )

    return tuple(support_terms)


def _build_claim_backstop_query_terms(debate_topic: str, transcript: list[dict]) -> tuple[str, ...]:
    """
    Build a fallback query that anchors to the debate topic and the assistant's
    most recent claim, rather than the user's exact evidence wording.
    """
    first_user = _first_user_context(transcript)
    assistant_claim = _assistant_context(transcript)
    parts = [debate_topic, first_user, assistant_claim]
    normalized_parts = [_normalize_space(part) for part in parts if _normalize_space(part)]
    combined = " ".join(normalized_parts)
    lowered = combined.lower()
    alias_groups = _matching_alias_groups(combined)

    terms: list[str] = []
    _append_unique_terms(terms, _topic_support_bundle(alias_groups, lowered))

    for alias_text in _expand_alias_terms(combined):
        _append_unique_terms(terms, _tokenize_query(alias_text))

    for part in normalized_parts:
        _append_unique_terms(terms, _tokenize_query(part))

    if "harm reduction" in lowered or "risk reduction" in lowered:
        _append_unique_terms(terms, ("overdose", "hiv", "hepatitis", "needle", "syringe"))
    if "equitable access" in lowered or "highest-risk groups" in lowered or "reach" in lowered:
        _append_unique_terms(terms, ("access", "reach", "barriers", "coverage"))

    return tuple(terms[:_MAX_QUERY_TERMS])


def _build_topic_backstop_query_terms(debate_topic: str, transcript: list[dict]) -> tuple[str, ...]:
    """
    Broad fallback for evidence requests: prefer topic-centric support pages
    rather than returning no concrete URL at all.
    """
    first_user = _first_user_context(transcript)
    combined = " ".join(part for part in (_normalize_space(debate_topic), _normalize_space(first_user)) if part)
    lowered = combined.lower()
    alias_groups = _matching_alias_groups(combined)

    terms: list[str] = []
    _append_unique_terms(terms, _topic_support_bundle(alias_groups, lowered))
    for alias_text in _expand_alias_terms(combined):
        _append_unique_terms(terms, _tokenize_query(alias_text))
    _append_unique_terms(terms, _tokenize_query(combined))
    return tuple(terms[:_MAX_QUERY_TERMS])


def _score_text(text: str, query_terms: tuple[str, ...]) -> int:
    lowered = (text or "").lower()
    if not lowered:
        return 0
    score = 0
    for term in query_terms:
        if term in lowered:
            term_score = max(1, min(len(term) - 2, 4))
            if term in _LOW_SIGNAL_QUERY_TERMS:
                term_score = 1
            elif term in _PRIORITY_QUERY_TERMS:
                term_score += 2
            score += term_score
    return score


def _matched_terms(text: str, query_terms: tuple[str, ...]) -> set[str]:
    lowered = (text or "").lower()
    return {term for term in query_terms if term in lowered}


def _page_penalty(url: str, title: str, required_alias_groups: tuple[tuple[str, ...], ...] = ()) -> int:
    lowered_url = (url or "").lower()
    lowered_title = (title or "").lower()
    text = f"{lowered_url} {lowered_title}"
    penalty = 0
    if "site index" in text or "indice del sitio" in text:
        penalty += 10
    if lowered_url.endswith("/site.html") or "/site.html" in lowered_url:
        penalty += 10
    if "sitemap" in text:
        penalty += 8
    if lowered_url.endswith("/index.html") or lowered_title == "index":
        penalty += 4
    if "/es/" in lowered_url:
        penalty += 2
    if _is_ssp_context(required_alias_groups):
        if "blood-disorders" in lowered_url or "blood disorders" in text:
            penalty += 24
        if "heavy-menstrual-bleeding" in lowered_url or "menstrual bleeding" in text:
            penalty += 20
    return penalty


def _anchor_terms(query_terms: tuple[str, ...]) -> set[str]:
    anchors = {term for term in query_terms if term not in _GENERIC_SIGNAL_TERMS}
    return {term for term in anchors if len(term) >= 4}


@lru_cache(maxsize=256)
def _fetch_url(url: str) -> tuple[str, str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=_REQUEST_TIMEOUT) as response:
            raw = response.read(_MAX_BYTES)
            content_type = response.headers.get_content_type() or ""
            charset = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            return response.geturl(), content_type, text
    except (HTTPError, URLError, TimeoutError, ValueError):
        return "", "", ""


def _extract_title(html_text: str, fallback_url: str) -> str:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        return _normalize_space(html.unescape(title_match.group(1)))
    parsed = urlparse(fallback_url)
    slug = parsed.path.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return _normalize_space(slug) or parsed.netloc


def _extract_description(html_text: str) -> str:
    meta_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if meta_match:
        return _normalize_space(html.unescape(meta_match.group(1)))
    return ""


def _extract_xml_locs(xml_text: str) -> list[str]:
    locs = re.findall(r"<loc>(.*?)</loc>", xml_text, flags=re.IGNORECASE | re.DOTALL)
    return [_normalize_space(html.unescape(loc)) for loc in locs if _normalize_space(loc)]


def _extract_homepage_links(page_url: str, html_text: str, allowed_netlocs: tuple[str, ...]) -> list[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html_text, flags=re.IGNORECASE)
    links = []
    for href in hrefs:
        absolute = urljoin(page_url, html.unescape(href))
        if _is_allowed_url(absolute, allowed_netlocs):
            links.append(absolute)
    return list(dict.fromkeys(links))


@lru_cache(maxsize=128)
def _discover_sitemap_urls(root_url: str) -> tuple[str, ...]:
    robots_url = f"{_domain_root(root_url)}/robots.txt"
    _, _, robots_text = _fetch_url(robots_url)
    sitemaps = []
    for line in robots_text.splitlines():
        if line.lower().startswith("sitemap:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                sitemaps.append(candidate)
    if not sitemaps:
        sitemaps.extend(
            [
                f"{_domain_root(root_url)}/sitemap.xml",
                f"{_domain_root(root_url)}/sitemap_index.xml",
                f"{_domain_root(root_url)}/sitemap-index.xml",
            ]
        )
    return tuple(dict.fromkeys(sitemaps))


def _rank_urls(
    urls: list[str],
    query_terms: tuple[str, ...],
    allowed_netlocs: tuple[str, ...],
    required_alias_groups: tuple[tuple[str, ...], ...] = (),
) -> list[tuple[int, str]]:
    ranked = []
    for url in urls:
        if not _is_allowed_url(url, allowed_netlocs):
            continue
        score = _score_text(url, query_terms)
        if required_alias_groups:
            if _matches_required_alias_groups(url, required_alias_groups):
                score += 12
            else:
                score -= 8
        score -= _page_penalty(url, url, required_alias_groups)
        ranked.append((score, url))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _crawl_sitemap(
    sitemap_url: str,
    query_terms: tuple[str, ...],
    allowed_netlocs: tuple[str, ...],
    required_alias_groups: tuple[tuple[str, ...], ...],
    depth: int,
    visited: set[str],
) -> list[str]:
    if depth > _MAX_SITEMAP_DEPTH or sitemap_url in visited:
        return []
    visited.add(sitemap_url)

    _, _, sitemap_text = _fetch_url(sitemap_url)
    locs = _extract_xml_locs(sitemap_text)
    if not locs:
        return []

    page_urls = []
    child_sitemaps = []
    for loc in locs:
        lowered = loc.lower()
        if lowered.endswith(".xml") or "sitemap" in lowered:
            child_sitemaps.append(loc)
        else:
            page_urls.append(loc)

    ranked_pages = _rank_urls(page_urls, query_terms, allowed_netlocs, required_alias_groups)
    good_pages = [url for score, url in ranked_pages if score > 0][: _MAX_SOURCE_CANDIDATES]
    if good_pages:
        return good_pages

    ranked_children = _rank_urls(child_sitemaps, query_terms, allowed_netlocs, required_alias_groups)
    selected_children = [url for score, url in ranked_children if score > 0][: _MAX_CHILD_SITEMAPS]
    if not selected_children:
        selected_children = child_sitemaps[:_MAX_CHILD_SITEMAPS]

    gathered = []
    for child_url in selected_children:
        gathered.extend(_crawl_sitemap(child_url, query_terms, allowed_netlocs, required_alias_groups, depth + 1, visited))
        if len(gathered) >= _MAX_SOURCE_CANDIDATES:
            break
    return list(dict.fromkeys(gathered))


def _discover_candidate_urls(
    source: dict,
    query_terms: tuple[str, ...],
    required_alias_groups: tuple[tuple[str, ...], ...] = (),
) -> list[str]:
    allowed_netlocs = _allowed_netlocs(source)
    candidates = []

    for root_url in source_url_list(source):
        for sitemap_url in _discover_sitemap_urls(root_url):
            candidates.extend(
                _crawl_sitemap(sitemap_url, query_terms, allowed_netlocs, required_alias_groups, 0, set())
            )
            if len(candidates) >= _MAX_SOURCE_CANDIDATES:
                break
        if candidates:
            break

    if not candidates:
        for root_url in source_url_list(source):
            final_url, _, html_text = _fetch_url(root_url)
            if not html_text:
                continue
            homepage_links = _extract_homepage_links(final_url or root_url, html_text, allowed_netlocs)
            ranked_links = _rank_urls(homepage_links, query_terms, allowed_netlocs, required_alias_groups)
            candidates.extend([url for score, url in ranked_links if score > 0][: _MAX_SOURCE_CANDIDATES])
            if candidates:
                break

    return list(dict.fromkeys(candidates))[:_MAX_SOURCE_CANDIDATES]


def _lookup_hits_for_terms(
    persona_id: str,
    query_terms: tuple[str, ...],
    required_alias_groups: tuple[tuple[str, ...], ...],
    max_hits: int,
    match_type: str,
) -> list[dict]:
    if not query_terms:
        return []
    anchor_terms = _anchor_terms(query_terms)

    ranked_sources = []
    for persona_rank, source in enumerate(relevant_credible_sources(persona_id), 1):
        metadata = " ".join([source["name"], source["specialty"], source["best_for"]])
        metadata_score = _score_text(metadata, query_terms)
        ranked_sources.append((metadata_score, persona_rank, source))
    ranked_sources.sort(key=lambda item: (-item[0], item[1]))

    hits = []
    for _, _, source in ranked_sources[:6]:
        candidates = _discover_candidate_urls(source, query_terms, required_alias_groups)
        for candidate_url in candidates[:_MAX_PAGE_FETCHES]:
            final_url, _, html_text = _fetch_url(candidate_url)
            if not html_text:
                continue
            title = _extract_title(html_text, final_url or candidate_url)
            description = _extract_description(html_text)
            combined_text = _normalize_space(f"{candidate_url} {title} {description}").lower()
            alias_group_match = _matches_required_alias_groups(combined_text, required_alias_groups)
            matched = (
                _matched_terms(candidate_url, query_terms)
                | _matched_terms(title, query_terms)
                | _matched_terms(description, query_terms)
            )
            anchor_matches = (
                _matched_terms(candidate_url, tuple(anchor_terms))
                | _matched_terms(title, tuple(anchor_terms))
                | _matched_terms(description, tuple(anchor_terms))
            ) if anchor_terms else set()
            score = (
                _score_text(candidate_url, query_terms)
                + _score_text(title, query_terms)
                + _score_text(description, query_terms)
                - _page_penalty(final_url or candidate_url, title, required_alias_groups)
            )
            lowered_result_url = (final_url or candidate_url).lower()
            lowered_title = title.lower()
            if (
                score <= 0
                or len(matched) < 2
                or (anchor_terms and not anchor_matches and not alias_group_match)
                or not alias_group_match
                or "site index" in lowered_title
                or lowered_result_url.endswith("/site.html")
            ):
                continue
            hits.append(
                {
                    "source_name": source["name"],
                    "url": final_url or candidate_url,
                    "title": title,
                    "description": description,
                    "score": score,
                }
            )

    hits.sort(key=lambda item: (-item["score"], item["source_name"], item["url"]))

    # Step 1: de-duplicate exact URLs while preserving score order.
    unique_hits = []
    seen_urls = set()
    for hit in hits:
        if hit["url"] in seen_urls:
            continue
        seen_urls.add(hit["url"])
        unique_hits.append(hit)

    # Step 2: prefer one hit per source bucket (source org/domain) when possible.
    selected_hits = []
    seen_buckets = set()
    for hit in unique_hits:
        bucket = _source_bucket(hit.get("source_name", ""), hit.get("url", ""))
        if bucket in seen_buckets:
            continue
        selected_hits.append(hit)
        seen_buckets.add(bucket)
        if len(selected_hits) >= max_hits:
            break

    # Step 3: fill remaining slots with best remaining hits from same buckets.
    if len(selected_hits) < max_hits:
        used_urls = {hit["url"] for hit in selected_hits}
        for hit in unique_hits:
            if hit["url"] in used_urls:
                continue
            selected_hits.append(hit)
            used_urls.add(hit["url"])
            if len(selected_hits) >= max_hits:
                break

    return [
        {
            "source_name": hit["source_name"],
            "url": hit["url"],
            "title": hit["title"],
            "description": hit["description"],
            "match_type": match_type,
        }
        for hit in selected_hits
    ]


def lookup_relevant_source_hits(
    persona_id: str,
    debate_topic: str,
    transcript: list[dict],
    max_hits: int = 3,
    force_fallback: bool = False,
) -> list[dict]:
    """
    Return concrete approved-domain page hits for an evidence request.

    Each hit includes:
        source_name, url, title, description, match_type
    """
    query_terms = _build_query_terms(debate_topic, transcript)
    required_alias_groups = _matching_alias_groups(" ".join(_query_context_parts(debate_topic, transcript)))

    hits = _lookup_hits_for_terms(
        persona_id,
        query_terms,
        required_alias_groups,
        max_hits,
        match_type="exact",
    )
    if hits:
        return hits

    latest_user = next((turn.get("content", "") for turn in reversed(transcript) if turn.get("role") == "user"), "")
    if not force_fallback and not is_evidence_request(latest_user):
        return []

    fallback_terms = _build_claim_backstop_query_terms(debate_topic, transcript)
    if fallback_terms and fallback_terms != query_terms:
        hits = _lookup_hits_for_terms(
            persona_id,
            fallback_terms,
            required_alias_groups,
            max_hits,
            match_type="closest",
        )
        if hits:
            return hits

    topic_backstop_terms = _build_topic_backstop_query_terms(debate_topic, transcript)
    if topic_backstop_terms and topic_backstop_terms not in (query_terms, fallback_terms):
        hits = _lookup_hits_for_terms(
            persona_id,
            topic_backstop_terms,
            required_alias_groups,
            max_hits,
            match_type="closest",
        )
        if hits:
            return hits

    # Last-resort evidence fallback: keep approved domains but relax strict
    # entity-group filtering so we can still return the closest policy pages.
    if required_alias_groups:
        for terms in (fallback_terms, topic_backstop_terms, query_terms):
            if not terms:
                continue
            hits = _lookup_hits_for_terms(
                persona_id,
                terms,
                (),
                max_hits,
                match_type="closest",
            )
            if hits:
                return hits

    return []
