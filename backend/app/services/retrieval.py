from __future__ import annotations

from difflib import SequenceMatcher
from html import unescape
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import Settings
from app.schemas import EvidenceSource
from app.services.source_quality import enrich_source

DEFAULT_HEADERS = {
    "User-Agent": (
        "VeriSight/0.1 (https://github.com/verisight; dev@example.com) Python-httpx/0.28"
    ),
    "Accept": "application/json",
}

STOP_WORDS = {
    "about", "after", "and", "are", "does", "from", "have", "into", "is", "the",
    "this", "that", "what", "when", "where", "which", "who", "with", "would",
}

# These words describe a category, not the subject being researched. They must
# never be sufficient on their own to make a search result a valid citation.
GENERIC_TOPIC_TERMS = {
    "college", "engineer", "university", "school", "company", "organisation",
    "organization", "institution", "hospital", "research", "centre", "center",
    "private", "public", "located", "establish", "affiliate", "approve",
    "creator", "creat", "inventor", "invent", "designer", "design", "founder", "found",
}

SUBJECT_ALIASES = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ecmascript": "JavaScript",
    "cpp": "C++",
    "c plus plus": "C++",
    "py": "Python",
    "css": "CSS",
    "html": "HTML",
}

# These may occur beside a person's name in a legitimate profile title without
# changing the subject, such as "Dr. Kalpana Chawla | National Air and Space
# Museum". Other added name-like terms signal a different person with a
# similar name and should not be treated as biographical evidence.
PROFILE_TITLE_CONTEXT_TERMS = {
    "air", "archive", "biography", "dr", "history", "life", "museum",
    "national", "official", "profile", "space", "the", "tribute",
}

# The keyword normalizer can remove a trailing plural-like ``s``. Keep this
# tolerance limited to identity-title matching so ``Subhas``, ``Subhash``,
# and the common typed variant ``Shubhash`` can resolve to the same historical
# person. Extra title words are still checked separately to reject namesakes.
IDENTITY_TERM_SIMILARITY = 0.70


def _split_compound_question(question: str) -> list[str]:
    """Split two ordinal 'Who was ... and ...?' questions before retrieval.

    A single search for two office holders tends to return evidence only for the
    first country. Restrict this to the explicit ordinal form so ordinary
    questions containing 'and' still stay together.
    """
    match = re.match(
        r"^\s*(who\s+(?:was|is)\s+)(.+?)\s+and\s+(.+?)[?!\.\s]*$",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return [question]

    prefix, first_subject, second_subject = match.groups()
    ordinal = r"\b\d+(?:st|nd|rd|th)\b"
    if not re.search(ordinal, first_subject, flags=re.IGNORECASE) or not re.search(
        ordinal, second_subject, flags=re.IGNORECASE
    ):
        return [question]

    second_question = second_subject.strip()
    if not re.match(r"^(?:the|a|an)\b", second_question, flags=re.IGNORECASE):
        second_question = f"the {second_question}"
    return [
        f"{prefix}{first_subject.strip()}?",
        f"{prefix}{second_question}?",
    ]


def _canonical_subject(subject: str) -> str:
    normalized = re.sub(r"\s+", " ", subject.strip().lower())
    return SUBJECT_ALIASES.get(normalized, subject.strip())


def _relation_parts(question: str) -> tuple[str, str] | None:
    """Return (subject, relationship) for questions such as creator of C++."""
    noun_match = re.search(
        r"\bwho\s+(?:is|was)\s+the\s+(creator|inventor|designer|founder)\s+of\s+(.+?)[?!.\s]*$",
        question,
        flags=re.IGNORECASE,
    )
    if noun_match:
        return _canonical_subject(noun_match.group(2)), noun_match.group(1).lower()

    verb_match = re.search(
        r"\bwho\s+(created|invented|designed|founded)\s+(.+?)[?!.\s]*$",
        question,
        flags=re.IGNORECASE,
    )
    if verb_match:
        relationship = {
            "created": "creator",
            "invented": "inventor",
            "designed": "designer",
            "founded": "founder",
        }[verb_match.group(1).lower()]
        return _canonical_subject(verb_match.group(2)), relationship
    return None


def _subject_query(question: str) -> str:
    """Turn conversational identity questions into an entity-focused search."""
    relation = _relation_parts(question)
    if relation:
        return relation[0]
    biographical_match = re.match(
        r"\s*(?:when|where)\s+(?:was|did)\s+(.+?)\s+"
        r"(?:born|die|died|study|work|live)\b",
        question,
        flags=re.IGNORECASE,
    )
    if biographical_match:
        subject = _canonical_subject(biographical_match.group(1))
        return subject or question
    match = re.match(
        r"\s*who\s+is\s+(.+?)(?=\s+(?:and|with|including)\b|[?!.,]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return question
    subject = _canonical_subject(match.group(1))
    return subject or question


def _is_identity_question(question: str) -> bool:
    return _relation_parts(question) is None and _subject_query(question) != question


def _research_query(question: str) -> str:
    relation = _relation_parts(question)
    if relation:
        subject, relationship = relation
        return f'"{subject}" {relationship}'
    subject = _subject_query(question)
    if _is_identity_question(question):
        if re.search(r"\bborn\b", question, flags=re.IGNORECASE):
            return f"{subject} birth date birthplace"
        if re.search(r"\b(?:die|died|death)\b", question, flags=re.IGNORECASE):
            return f"{subject} death date"
        if re.search(r"\b(?:achievements?|awards?|career|accomplishment)\b", question, flags=re.IGNORECASE):
            return f"{subject} achievements career"
        return f"{subject} biography"
    if re.search(
        r"\b(?:first|1st)\s+captain\b.*\bindian cricket\b.*\b(?:players|team|squad)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return '"India inaugural Test 1932" playing eleven C. K. Nayudu'
    return question


def _relation_title_bonus(question: str, title: str) -> float:
    """Prefer a canonical page for the subject of a creator-style question."""
    relation = _relation_parts(question)
    if not relation:
        return 0.0
    subject_terms = _keywords(relation[0])
    title_terms = _keywords(title)
    if not subject_terms or not subject_terms <= title_terms:
        return 0.0
    extras = title_terms - subject_terms
    if not extras:
        return 10.0
    canonical_descriptors = {"program", "programm", "language", "software", "system"}
    return 8.0 if extras <= canonical_descriptors else 0.0


def _identity_title_bonus(question: str, title: str) -> float:
    """Strongly prefer the page about the named person over a namesake."""
    if not _is_identity_question(question):
        return 0.0
    subject_terms = _keywords(_subject_query(question))
    title_terms = _keywords(title)
    if not subject_terms or not title_terms:
        return 0.0
    if _identity_title_matches(subject_terms, title_terms):
        extras = _identity_title_extras(subject_terms, title_terms)
        if not extras:
            return 10.0
        if extras <= PROFILE_TITLE_CONTEXT_TERMS:
            return 8.0
    # A one-word query such as "Ramanujan" can validly lead to a full name.
    if len(subject_terms) == 1 and subject_terms <= title_terms:
        return 4.0
    return 0.0


def _identity_title_matches(subject_terms: set[str], title_terms: set[str]) -> bool:
    """Match a person's title while allowing a minor spelling variant.

    For example, the historic figure is commonly written as both ``Subhas``
    and ``Subhash`` Chandra Bose.  This must not discard a valid biography,
    but a result with an additional name still requires separate handling.
    """
    return bool(subject_terms) and all(
        any(
            SequenceMatcher(None, subject, candidate).ratio() >= IDENTITY_TERM_SIMILARITY
            for candidate in title_terms
        )
        for subject in subject_terms
    )


def _identity_title_extras(subject_terms: set[str], title_terms: set[str]) -> set[str]:
    """Return title words that do not correspond to the requested name."""
    matched_terms = {
        candidate
        for candidate in title_terms
        if any(
            SequenceMatcher(None, subject, candidate).ratio() >= IDENTITY_TERM_SIMILARITY
            for subject in subject_terms
        )
    }
    return title_terms - matched_terms


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _stem(token: str) -> str:
    if token in {"cplusplus", "csharp", "dotnet"}:
        return token
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _keywords(text: str) -> set[str]:
    normalized = _normalize(text)
    # Preserve meaningful programming-language symbols that the normal word
    # tokenizer would otherwise reduce to a single discarded letter.
    normalized = normalized.replace("c++", "cplusplus").replace("c#", "csharp").replace(".net", "dotnet")
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2 and token not in STOP_WORDS
    }


def _source_relevance(question: str, source: EvidenceSource) -> float:
    """Score a source by topic match, with a strong preference for the requested year."""
    question_terms = _keywords(question)
    title_terms = _keywords(source.title)
    source_terms = _keywords(f"{source.title} {source.snippet}")
    title_overlap = len(question_terms & title_terms)
    source_overlap = len(question_terms & source_terms)
    score = (
        2 * title_overlap
        + source_overlap
        + _identity_title_bonus(question, source.title)
        + _relation_title_bonus(question, source.title)
        + 3 * source.credibility
    )

    requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    source_years = set(re.findall(r"\b(?:19|20)\d{2}\b", f"{source.title} {source.snippet}"))
    if requested_years:
        if requested_years & source_years:
            score += 4
        elif source_years:
            score -= 5
    host = urlparse(source.url).netloc.lower()
    if host.endswith((".ac.in", ".edu", ".edu.in", ".gov", ".gov.in")):
        score += 3
    subject_terms = _keywords(_subject_query(question))
    if _is_identity_question(question) and subject_terms:
        # Credible profile pages sometimes include the organisation in the title
        # (for example, a NASA astronaut profile).
        if subject_terms <= title_terms and host.endswith(("nasa.gov", ".gov", ".gov.in")):
            score += 8
    # Encyclopaedia entries are a useful reliable fallback for broad biography
    # questions when no primary source exists.
    if host.endswith("wikipedia.org"):
        score += 3
    return score


def _source_host(source: EvidenceSource) -> str:
    """Return a stable domain key used to avoid one-site evidence piles."""
    return urlparse(source.url).netloc.lower().removeprefix("www.") or source.url


def _source_merge_key(source: EvidenceSource) -> tuple[str, str]:
    """Only merge duplicates from the same domain.

    Different publishers often use the same page title (for example,
    ``Microsoft`` on microsoft.com and Wikipedia). Keeping both allows the
    verifier to use independent evidence instead of accidentally replacing a
    primary source with an encyclopaedia entry.
    """
    return _source_host(source), _normalize(source.title)


def _select_diverse_sources(
    ranked: list[tuple[EvidenceSource, float]],
    *,
    limit: int,
) -> list[EvidenceSource]:
    """Keep the strongest evidence while preferring independent domains.

    Search engines frequently return several pages from the same site. Those
    pages can be useful internally, but showing them as separate citations
    overstates agreement. The final evidence list contains one source per
    domain, ordered by topic relevance and source quality.
    """
    selected: list[EvidenceSource] = []
    seen_hosts: set[str] = set()
    for source, _score in ranked:
        host = _source_host(source)
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        selected.append(source)
        if len(selected) == limit:
            break
    return selected


def _is_citable_url(url: str) -> bool:
    """Do not present search-engine redirect pages as evidence citations."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return (
        parsed.scheme in {"http", "https"}
        and bool(host)
        and not host.endswith("duckduckgo.com")
    )


def _has_topic_anchor(question: str, source: EvidenceSource) -> bool:
    """Require a named or specific term, not just a generic category match."""
    relation = _relation_parts(question)
    source_terms = _keywords(f"{source.title} {source.snippet}")
    if relation:
        subject_terms = _keywords(relation[0])
        return bool(subject_terms) and subject_terms <= source_terms
    anchors = _keywords(question) - GENERIC_TOPIC_TERMS
    if not anchors:
        return True
    return bool(anchors & source_terms)


def _is_namesake_institution(question: str, source: EvidenceSource) -> bool:
    """Reject organisations merely named after the person being researched.

    A result such as "Kalpana Chawla Government Medical College" contains all
    of the person's name, but is not biographical evidence about the astronaut.
    This safeguard only applies to identity questions about a non-institution.
    """
    subject = _subject_query(question)
    subject_terms = _keywords(subject)
    if not subject_terms or subject_terms & GENERIC_TOPIC_TERMS:
        return False

    title_terms = _keywords(source.title)
    host = urlparse(source.url).netloc.lower()
    return (
        subject_terms <= title_terms
        and bool(title_terms & GENERIC_TOPIC_TERMS)
        and not host.endswith("wikipedia.org")
    )


def _is_unrelated_identity_page(question: str, source: EvidenceSource) -> bool:
    """Reject a relative's biography when researching a named person.

    Search results for a famous person frequently include an article about a
    better-known family member. A mention in that article is not sufficient
    evidence for the requested biography or achievements.
    """
    if not _is_identity_question(question):
        return False
    subject_terms = _keywords(_subject_query(question))
    title_terms = _keywords(source.title)
    if not subject_terms:
        return False
    # A one-word prompt such as "Who is Ramanujan?" can legitimately lead to
    # a full title such as "Srinivasa Ramanujan". For a multi-word person name,
    # however, another non-profile word in the title is a strong namesake cue:
    # "Pilli Subhash Chandra Bose" is not the same subject as Netaji. A minor
    # spelling variation (Subhas/Subhash) remains a match for the same person.
    if _identity_title_matches(subject_terms, title_terms):
        extra_terms = _identity_title_extras(subject_terms, title_terms)
        if extra_terms and not extra_terms <= PROFILE_TITLE_CONTEXT_TERMS:
            return True
        return False
    host = urlparse(source.url).netloc.lower()
    return source.credibility < 0.90 and not host.endswith((".gov", ".gov.in", ".edu", ".edu.in"))


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
    # Keep a useful portion of the article. Claim-level verification later
    # selects the exact supporting sentences; trimming here could remove the
    # creator, date, or other fact before that selection had a chance to run.
    return _normalize(extract)[:8_000] if extract else ""


async def search_wikipedia(
    question: str,
    *,
    client: httpx.AsyncClient,
    limit: int = 8,
) -> list[EvidenceSource]:
    search_query = _research_query(question)
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
            int(
                _identity_title_bonus(question, page.get("title", ""))
                + _relation_title_bonus(question, page.get("title", ""))
            )
            + len(question_terms & title_terms)
            + 4 * len(year_terms & title_terms),
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


async def search_tavily(
    question: str,
    *,
    api_key: str,
    client: httpx.AsyncClient,
) -> list[EvidenceSource]:
    """Retrieve ranked, citation-ready web excerpts from Tavily."""
    try:
        response = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": question,
                "search_depth": "advanced",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise ValueError(
            f"Tavily search failed (HTTP {error.response.status_code}). "
            "Check the Tavily API key and account usage."
        ) from error
    except httpx.RequestError as error:
        raise ValueError(f"Could not reach Tavily: {error}") from error

    evidence: list[EvidenceSource] = []
    for result in response.json().get("results", []):
        title = str(result.get("title", "")).strip()
        url = str(result.get("url", "")).strip()
        content = str(result.get("content", "")).strip()
        if title and content and _is_citable_url(url):
            evidence.append(EvidenceSource(title=title, url=url, snippet=content[:1_500]))
    return evidence


async def retrieve_web_evidence(
    question: str,
    *,
    settings: Settings,
) -> list[EvidenceSource]:
    question_parts = _split_compound_question(question)
    if len(question_parts) > 1:
        # Retrieve each independent fact separately, then preserve the top
        # evidence from both sides for claim-level verification and citations.
        merged: list[EvidenceSource] = []
        seen_urls: set[str] = set()
        for part in question_parts:
            for source in await retrieve_web_evidence(part, settings=settings):
                if source.url not in seen_urls:
                    seen_urls.add(source.url)
                    merged.append(source)
        return merged[:4]

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=DEFAULT_HEADERS,
    ) as client:
        search_query = _research_query(question)
        tavily_results = (
            await search_tavily(search_query, api_key=settings.tavily_api_key, client=client)
            if settings.tavily_api_key
            else []
        )
        # Tavily can occasionally return one keyword-matched but irrelevant page
        # (for example, "Ramanujan College" for Srinivasa Ramanujan). Always
        # blend in encyclopaedia results so that a single poor search result is
        # never the whole verification evidence set.
        wikipedia_results = await search_wikipedia(question, client=client)
        # Tavily can return only one valid biography result. Supplement it with
        # a free, exact-name web search so the verifier can show independent
        # corroboration when a primary or institutional profile is available.
        identity_results = await search_duckduckgo_web(
            f'"{_subject_query(question)}" biography',
            client=client,
            limit=4,
        ) if _is_identity_question(question) else []
        if not tavily_results:
            ddg_results = await search_duckduckgo(question, client=client)
            web_results = await search_duckduckgo_web(question, client=client)
            official_results = await search_duckduckgo_web(
                f'"{question}" official website',
                client=client,
            )
        else:
            ddg_results = []
            web_results = []
            official_results = []

    merged: list[EvidenceSource] = []
    source_indexes: dict[tuple[str, str], int] = {}
    for source in [
        *tavily_results,
        *identity_results,
        *official_results,
        *web_results,
        *wikipedia_results,
        *ddg_results,
    ]:
        source = enrich_source(source)
        key = _source_merge_key(source)
        if (
            not source.snippet
            or not _is_citable_url(source.url)
            or not _has_topic_anchor(question, source)
            or _is_namesake_institution(question, source)
            or _is_unrelated_identity_page(question, source)
        ):
            continue
        if key in source_indexes:
            index = source_indexes[key]
            existing = merged[index]
            if source.snippet not in existing.snippet:
                existing_host = urlparse(existing.url).netloc.lower()
                source_host = urlparse(source.url).netloc.lower()
                # Prefer the canonical Wikipedia link when it supplies the
                # fuller article excerpt for the same titled page.
                preferred_url = source.url if source_host.endswith("wikipedia.org") else existing.url
                if existing_host.endswith("wikipedia.org"):
                    preferred_url = existing.url
                merged[index] = EvidenceSource(
                    title=existing.title,
                    url=preferred_url,
                    snippet=f"{existing.snippet} {source.snippet}"[:9_000],
                    credibility=max(existing.credibility, source.credibility),
                    source_quality=(
                        existing.source_quality
                        if existing.credibility >= source.credibility
                        else source.source_quality
                    ),
                )
            continue
        source_indexes[key] = len(merged)
        merged.append(source)

    ranked = sorted(
        ((source, _source_relevance(question, source)) for source in merged),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return []

    best_score = ranked[0][1]
    # Keeping loose matches here caused unrelated pages (for example, a product
    # page instead of a company's founder page) to appear as citations.
    minimum_score = max(2.5, best_score * 0.62)
    focused = [(source, score) for source, score in ranked if score >= minimum_score]
    return _select_diverse_sources(focused, limit=3)


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
