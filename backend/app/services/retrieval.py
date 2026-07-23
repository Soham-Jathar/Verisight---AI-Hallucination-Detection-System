from __future__ import annotations

from html import unescape
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import Settings
from app.schemas import EvidenceSource

DEFAULT_HEADERS = {
    "User-Agent": (
        "VeriSight/0.1 (https://github.com/verisight; dev@example.com) Python-httpx/0.28"
    ),
    "Accept": "application/json",
}

STOP_WORDS = {
    "about", "after", "are", "does", "from", "have", "into", "is", "the",
    "this", "that", "what", "when", "where", "which", "who", "with", "would",
}


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _stem(token: str) -> str:
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _keywords(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(token) > 2 and token not in STOP_WORDS
    }


def _source_relevance(question: str, source: EvidenceSource) -> float:
    """Score a source by topic match, with a strong preference for the requested year."""
    question_terms = _keywords(question)
    title_terms = _keywords(source.title)
    source_terms = _keywords(f"{source.title} {source.snippet}")
    title_overlap = len(question_terms & title_terms)
    source_overlap = len(question_terms & source_terms)
    score = 2 * title_overlap + source_overlap

    requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    source_years = set(re.findall(r"\b(?:19|20)\d{2}\b", f"{source.title} {source.snippet}"))
    if requested_years:
        if requested_years & source_years:
            score += 4
        elif source_years:
            score -= 5
    return score


def _select_relevant_sentences(
    text: str,
    question: str,
    title: str,
    *,
    limit: int = 3,
) -> str:
    """Keep a compact, question-relevant excerpt from a full article."""
    sentences = re.split(r"(?<=[.!?])\s+", _normalize(text))
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 35]
    if not sentences:
        return _normalize(text)[:1_200]

    query_terms = _keywords(question) | _keywords(title)
    needs_winner = bool({"won", "winner", "winning", "champion", "champions"} & _keywords(question))

    ranked: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _keywords(sentence)
        overlap = len(query_terms & sentence_terms)
        result_bonus = 2 if needs_winner and {"won", "winner", "champion", "defeated"} & sentence_terms else 0
        ranked.append((overlap + result_bonus, index, sentence))

    best = sorted(ranked, reverse=True)[:limit]
    selected = [sentence for _, _, sentence in sorted(best, key=lambda item: item[1])]
    return " ".join(selected)


async def _article_excerpt(
    title: str,
    question: str,
    *,
    client: httpx.AsyncClient,
) -> str:
    """Fetch full plain-text article content, then retain only useful evidence."""
    try:
        response = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
                "titles": title,
            },
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return ""

    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = page.get("extract", "")
    return _select_relevant_sentences(extract, question, title) if extract else ""


async def search_wikipedia(
    question: str,
    *,
    client: httpx.AsyncClient,
    limit: int = 8,
) -> list[EvidenceSource]:
    search_query = question
    capital_match = re.search(r"\bcapital of\s+(.+?)\??$", question, flags=re.IGNORECASE)
    ceo_match = re.search(r"\bceo of\s+(.+?)\??$", question, flags=re.IGNORECASE)
    creator_match = re.search(r"\bwho created\s+(.+?)\??$", question, flags=re.IGNORECASE)
    founder_match = re.search(r"\bfounder of\s+(.+?)\??$", question, flags=re.IGNORECASE)

    if capital_match:
        search_query = f"{capital_match.group(1).strip()} capital city"
    elif ceo_match:
        search_query = f"{ceo_match.group(1).strip()} CEO"
    elif creator_match:
        search_query = f"{creator_match.group(1).strip()} creator"
    elif founder_match:
        search_query = f"{founder_match.group(1).strip()} founders"
    elif re.search(r"\b(who|which team)\s+won\b", question, flags=re.IGNORECASE):
        # Result questions are usually answered on a competition's final-match page.
        search_query = f"{question} final"

    try:
        search_response = await client.get(
            "https://en.wikipedia.org/w/rest.php/v1/search/page",
            params={"q": search_query, "limit": limit},
        )
        search_response.raise_for_status()
    except httpx.HTTPError:
        return []

    pages = search_response.json().get("pages", [])
    question_terms = _keywords(question)
    year_terms = {term for term in question_terms if term.isdigit() and len(term) == 4}

    def page_score(page: dict) -> tuple[int, int]:
        title_terms = _keywords(page.get("title", ""))
        return (
            len(question_terms & title_terms) + 4 * len(year_terms & title_terms),
            len(year_terms & title_terms),
        )

    pages = sorted(pages, key=page_score, reverse=True)[:3]
    evidence: list[EvidenceSource] = []

    for page in pages:
        title = page.get("title")
        page_key = page.get("key")
        if not title or not page_key:
            continue

        snippet = _strip_html(page.get("excerpt", "")).strip()
        page_url = f"https://en.wikipedia.org/wiki/{page_key.replace(' ', '_')}"

        try:
            summary_response = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_key}"
            )
            if summary_response.status_code == 200:
                payload = summary_response.json()
                snippet = payload.get("extract") or snippet
                page_url = (
                    payload.get("content_urls", {}).get("desktop", {}).get("page", page_url)
                )
        except httpx.HTTPError:
            pass

        summary_snippet = snippet
        article_excerpt = await _article_excerpt(title, question, client=client)
        if article_excerpt:
            # The lead summary often states the basic fact; the excerpt adds detail.
            snippet = f"{summary_snippet} {article_excerpt}".strip()

        if not snippet:
            continue

        evidence.append(
            EvidenceSource(
                title=title,
                url=page_url,
                snippet=snippet,
            )
        )

    return evidence


async def search_duckduckgo(
    question: str,
    *,
    client: httpx.AsyncClient,
) -> list[EvidenceSource]:
    try:
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": question, "format": "json", "no_redirect": 1, "no_html": 1},
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    payload = response.json()

    evidence: list[EvidenceSource] = []
    abstract = payload.get("AbstractText", "").strip()
    if abstract:
        evidence.append(
            EvidenceSource(
                title=payload.get("Heading") or "DuckDuckGo Instant Answer",
                url=payload.get("AbstractURL") or "",
                snippet=abstract,
            )
        )

    for topic in payload.get("RelatedTopics", [])[:3]:
        if not isinstance(topic, dict):
            continue
        text = topic.get("Text", "").strip()
        if not text:
            continue
        evidence.append(
            EvidenceSource(
                title=text.split(" - ", 1)[0],
                url=topic.get("FirstURL", ""),
                snippet=text,
            )
        )

    return evidence


def _direct_search_url(url: str) -> str:
    """Convert DuckDuckGo's redirect link into the original citation URL."""
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlparse(url)
    destination = parse_qs(parsed.query).get("uddg", [""])[0]
    return unquote(destination) or url


async def search_duckduckgo_web(
    question: str,
    *,
    client: httpx.AsyncClient,
    limit: int = 5,
) -> list[EvidenceSource]:
    """Use public search-result snippets when an instant answer is unavailable."""
    try:
        response = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": question},
            headers={
                "User-Agent": DEFAULT_HEADERS["User-Agent"],
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    links = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        response.text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        response.text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    evidence: list[EvidenceSource] = []
    for index, (url, raw_title) in enumerate(links[:limit]):
        title = _strip_html(raw_title)
        snippet = _strip_html(snippets[index]) if index < len(snippets) else ""
        direct_url = _direct_search_url(url)
        if title and snippet and direct_url.startswith(("http://", "https://")):
            evidence.append(EvidenceSource(title=title, url=direct_url, snippet=snippet))

    return evidence


async def retrieve_web_evidence(
    question: str,
    *,
    settings: Settings,
) -> list[EvidenceSource]:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=DEFAULT_HEADERS,
    ) as client:
        wikipedia_results = await search_wikipedia(question, client=client)
        ddg_results = await search_duckduckgo(question, client=client)
        web_results = await search_duckduckgo_web(question, client=client)

    merged: list[EvidenceSource] = []
    seen_titles: set[str] = set()
    for source in [*wikipedia_results, *ddg_results, *web_results]:
        key = _normalize(source.title)
        if not source.snippet or key in seen_titles:
            continue
        seen_titles.add(key)
        merged.append(source)

    ranked = sorted(
        ((source, _source_relevance(question, source)) for source in merged),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return []

    best_score = ranked[0][1]
    minimum_score = max(1.0, best_score * 0.45)
    focused = [source for source, score in ranked if score >= minimum_score]
    return focused[:2]


def build_evidence_answer(question: str, evidence: list[EvidenceSource]) -> str:
    if not evidence:
        return (
            f"I could not retrieve web evidence for '{question}'. "
            "Try rephrasing the question or configure an LLM provider for generation."
        )

    lead = evidence[0]
    supporting = evidence[1:2]
    answer_parts = [lead.snippet.rstrip(".") + "."]
    for source in supporting:
        answer_parts.append(source.snippet.rstrip(".") + ".")
    return " ".join(answer_parts)
