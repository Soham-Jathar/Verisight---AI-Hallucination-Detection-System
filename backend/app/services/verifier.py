from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
import re
import unicodedata

from app.schemas import ClaimAssessment, EvidenceSource

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
SEMANTIC_RERANKER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class NLIUnavailable(RuntimeError):
    """Raised when the optional local NLI model cannot be loaded."""


@lru_cache
def _semantic_reranker():
    """Load a compact local embedding model only when sentence reranking is used."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(SEMANTIC_RERANKER_MODEL)
    except Exception:
        return None


def _semantic_sentence_scores(claim: str, sentences: list[str]) -> list[float] | None:
    """Return cosine similarities, or fall back cleanly when embeddings are unavailable."""
    model = _semantic_reranker()
    if model is None or not sentences:
        return None
    try:
        vectors = model.encode([claim, *sentences], normalize_embeddings=True)
        claim_vector = vectors[0]
        return [float(claim_vector @ vector) for vector in vectors[1:]]
    except Exception:
        return None


def _normalize(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text.strip().lower())


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[a-z0-9]+", _normalize(left)) if len(token) > 2}
    right_tokens = {token for token in re.findall(r"[a-z0-9]+", _normalize(right)) if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _evidence_match(claim: str, source: EvidenceSource) -> float:
    """Lightweight relevance score used for selecting citations and fallback only."""
    claim_tokens = {token for token in re.findall(r"[a-z0-9]+", _normalize(claim)) if len(token) > 2}
    evidence_text = f"{source.title} {source.snippet}"
    evidence_tokens = {token for token in re.findall(r"[a-z0-9]+", _normalize(evidence_text)) if len(token) > 2}
    if not claim_tokens or not evidence_tokens:
        return 0.0
    coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    return 0.8 * coverage + 0.2 * _token_overlap(claim, evidence_text)


def _source_host(source: EvidenceSource) -> str:
    """Use a domain key to prevent duplicate citations from one website."""
    match = re.search(r"^[a-z]+://(?:www\.)?([^/]+)", source.url, flags=re.IGNORECASE)
    return match.group(1).lower() if match else source.url


def _entity_tokens(text: str) -> tuple[list[str], str]:
    """Return name-like terms and initials for direct page-title matching."""
    raw_tokens = re.findall(r"[a-z0-9]+", _normalize(text))
    terms = [token for token in raw_tokens if len(token) > 1]
    # Keep one-letter components here: ``R. D. Burman`` and ``Rahul Dev
    # Burman`` both become ``rdb``.
    initials = "".join(token[0] for token in raw_tokens if token[0].isalpha())
    return terms, initials


def _claim_subject(claim: str) -> str:
    """Extract the entity named at the start of a factual claim."""
    match = re.match(
        r"^(.+?)\s+(?:is|was|are|were|served|created|founded|led|became|"
        r"received|earned|wrote|died|passed|played|worked|born)\b",
        claim,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    # Answers to yes/no questions often begin with "No,". It is not part of
    # the entity and must not weaken entity alignment for the actual subject.
    return re.sub(r"^(?:no|yes)[,\s]+", "", match.group(1).strip(), flags=re.IGNORECASE)


def _entity_title_alignment(claim: str, title: str) -> float:
    """Prefer citations whose title is directly about the claim's entity.

    This distinguishes a canonical page such as ``R. D. Burman`` from a
    related page such as ``Meera Dev Burman``. It also handles initials such
    as ``A. P. J.`` without hard-coding any individual name.
    """
    subject = _claim_subject(claim)
    subject_terms, subject_initials = _entity_tokens(subject)
    title_terms, title_initials = _entity_tokens(title)
    if not subject_terms or not title_terms:
        return 0.0

    subject_set = set(subject_terms)
    title_set = set(title_terms)
    if subject_set <= title_set:
        return 1.0 if subject_set == title_set else 0.85
    if subject_initials and subject_initials == title_initials:
        return 0.95

    overlap = len(subject_set & title_set) / len(subject_set)
    return overlap * 0.45


def _entity_text_alignment(claim: str, text: str) -> float:
    """Measure whether a source is actually about the claim's named subject.

    This is deliberately tolerant of ordinary spelling variants (for example
    ``Subhas``/``Subhash``) and initials, but a page that merely shares a
    generic word such as ``India`` or ``history`` receives no entity credit.
    """
    subject = _claim_subject(claim)
    subject_terms, subject_initials = _entity_tokens(subject)
    text_terms, text_initials = _entity_tokens(text)
    if not subject_terms or not text_terms:
        return 0.0
    if subject_initials and subject_initials == text_initials:
        return 1.0

    matches = sum(
        any(SequenceMatcher(None, term, candidate).ratio() >= 0.82 for candidate in text_terms)
        for term in set(subject_terms)
    )
    return matches / len(set(subject_terms))


def _is_indirect_collection_source(claim: str, source: EvidenceSource) -> bool:
    """Reject generic list pages when their title is not about the claim entity.

    A list of aircraft incidents can mention a historic figure, yet it is not
    a suitable biography citation or the sole basis for contradicting a claim.
    Direct list pages such as ``List of awards received by X`` are retained
    because their title still identifies the requested subject.
    """
    if not _claim_subject(claim):
        return False
    generic_collection = re.match(
        r"^\s*(?:list|timeline|history|category|index|outline)\s+(?:of|for)\b",
        _normalize(source.title),
    )
    return bool(generic_collection and _entity_title_alignment(claim, source.title) < 0.45)


def _rank_claim_sources(
    claim: str,
    evidence: list[EvidenceSource],
    *,
    limit: int,
    minimum_score: float,
) -> list[tuple[float, EvidenceSource]]:
    """Rank candidates by claim match, source quality, and domain diversity."""
    ranked: list[tuple[float, EvidenceSource]] = []
    for source in evidence:
        source_text = f"{source.title} {source.snippet}"
        entity_alignment = _entity_text_alignment(claim, source_text)
        if _is_indirect_collection_source(claim, source):
            continue
        # For a named subject, avoid allowing an unrelated page with a few
        # generic words in common to become the claim's only evidence. Pronoun
        # and list-item claims have no extractable subject and remain eligible.
        if _claim_subject(claim) and entity_alignment < 0.67:
            continue
        score = (
            _evidence_match(claim, source) * (0.75 + 0.25 * source.credibility)
            + 0.25 * _entity_title_alignment(claim, source.title)
            + 0.10 * entity_alignment
        )
        ranked.append((score, source))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, EvidenceSource]] = []
    seen_hosts: set[str] = set()
    for score, source in ranked:
        if score < minimum_score:
            continue
        host = _source_host(source)
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        selected.append((score, source))
        if len(selected) == limit:
            break
    return selected


def _lexical_support(claim: str, evidence: list[EvidenceSource]) -> float:
    """Conservative fallback for an NLI-neutral claim quoted almost verbatim."""
    claim_tokens = {
        token for token in re.findall(r"[a-z0-9]+", _normalize(claim)) if len(token) > 2
    }
    if not claim_tokens:
        return 0.0

    numeric_claim_tokens = {
        token for token in claim_tokens
        if token.isdigit() or token in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
    }
    best = 0.0
    for source in evidence:
        evidence_tokens = set(re.findall(r"[a-z0-9]+", _normalize(f"{source.title} {source.snippet}")))
        if numeric_claim_tokens and not numeric_claim_tokens <= evidence_tokens:
            continue
        coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        best = max(best, coverage)
    return best


def _strong_textual_support(claim: str, evidence: list[EvidenceSource]) -> float:
    """Recognise a near-verbatim fact before trusting one NLI outlier.

    Dates and other numeric terms are already required by ``_lexical_support``.
    The additional polarity check prevents a claim such as "did not serve" from
    being treated as support merely because all the surrounding words match.
    """
    negation_terms = {"not", "never", "neither", "no", "without"}
    claim_terms = set(re.findall(r"[a-z0-9]+", _normalize(claim)))
    claim_is_negative = bool(claim_terms & negation_terms)
    best = 0.0
    for source in evidence:
        excerpt = _claim_evidence_excerpt(claim, source)
        source_terms = set(re.findall(r"[a-z0-9]+", _normalize(excerpt)))
        if bool(source_terms & negation_terms) != claim_is_negative:
            continue
        best = max(best, _lexical_support(claim, [source]))
    return best


def _collapses_disputed_report_into_fact(claim: str, evidence: list[EvidenceSource]) -> bool:
    """Detect an answer that removes an important uncertainty attribution.

    A search result can say that a newspaper *reported* an event while an
    authority says the report is unverified or wrong.  Repeating the event as
    an established fact is not evidence-grounded, even when many content words
    overlap.  This rule is intentionally narrow: it requires a factual event,
    a sufficiently related source, both report-attribution and denial language,
    and no uncertainty marker in the generated claim itself.
    """
    normalized_claim = _normalize(claim)
    if re.search(r"\b(?:reportedly|allegedly|according to|said to|claimed)\b", normalized_claim):
        return False
    if not re.search(
        r"\b(?:recovered|found|confirmed|proven|established|occurred|caused|responsible|exists)\b",
        normalized_claim,
    ):
        return False

    attribution = re.compile(r"\b(?:reported|reports|claimed|claims|alleged|purported|according to|source said)\b")
    denial = re.compile(
        r"\b(?:not aware|no (?:evidence|confirmation|proof)|not confirmed|unverified|"
        r"completely wrong|unwarranted|denied|hadn't|hasn't|have not|has not)\b"
    )
    return any(
        attribution.search(_normalize(source.snippet))
        and denial.search(_normalize(source.snippet))
        and _evidence_match(claim, source) >= 0.45
        for source in evidence
    )


_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}


def _date_signatures(text: str) -> set[str]:
    """Normalize written and ISO dates so harmless formatting does not mislead NLI."""
    normalized = _normalize(text)
    signatures = {
        f"{year}-{month}-{day}"
        for year, month, day in re.findall(r"\b((?:1[5-9]|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", normalized)
    }
    for day, month_name, year in re.findall(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\s+((?:1[5-9]|20)\d{2})\b",
        normalized,
    ):
        signatures.add(f"{year}-{_MONTHS[month_name]}-{int(day):02d}")
    for month_name, day, year in re.findall(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+((?:1[5-9]|20)\d{2})\b",
        normalized,
    ):
        signatures.add(f"{year}-{_MONTHS[month_name]}-{int(day):02d}")
    return signatures


def _has_direct_date_support(claim: str, evidence: list[EvidenceSource]) -> bool:
    """Accept an exact, entity-aligned date before an NLI outlier can reject it."""
    claim_dates = _date_signatures(claim)
    if not claim_dates:
        return False

    claim_terms = {
        token for token in re.findall(r"[a-z0-9]+", _normalize(claim))
        if len(token) > 2 and token not in _MONTHS and not token.isdigit()
    }
    for source in evidence:
        entity_alignment = _entity_text_alignment(claim, f"{source.title} {source.snippet}")
        for sentence in re.split(r"(?<=[.!?;])\s+", source.snippet):
            if not (claim_dates & _date_signatures(sentence)):
                continue
            sentence_terms = set(re.findall(r"[a-z0-9]+", _normalize(sentence)))
            overlap = len(claim_terms & sentence_terms) / len(claim_terms) if claim_terms else 0.0
            if entity_alignment >= 0.67 or overlap >= 0.65:
                return True
    return False


def _has_direct_year_support(claim: str, evidence: list[EvidenceSource]) -> bool:
    """Recognise an exact, entity-aligned year for the same factual event."""
    claim_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", _normalize(claim)))
    if not claim_years:
        return False
    ignored = claim_years | {"the", "was", "were", "is", "are", "and", "for", "with"}
    claim_terms = {
        token for token in re.findall(r"[a-z0-9]+", _normalize(claim))
        if len(token) > 2 and token not in ignored
    }
    for source in evidence:
        entity_alignment = _entity_text_alignment(claim, f"{source.title} {source.snippet}")
        if entity_alignment < 0.67:
            continue
        for sentence in re.split(r"(?<=[.!?;])\s+", source.snippet):
            sentence_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", _normalize(sentence)))
            if not (claim_years & sentence_years):
                continue
            sentence_terms = set(re.findall(r"[a-z0-9]+", _normalize(sentence)))
            overlap = len(claim_terms & sentence_terms) / len(claim_terms) if claim_terms else 0.0
            if overlap >= 0.55:
                return True
    return False


def _has_negative_year_fact_support(claim: str, evidence: list[EvidenceSource]) -> bool:
    """Support claims such as 'Einstein was not born in 1889' from the true year.

    A different source year contradicts a positive claim, but it *supports* a
    negated year claim. Treating both forms identically was the reason valid
    false-premise answers could receive an unsupported verdict.
    """
    normalized_claim = _normalize(claim)
    claim_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", normalized_claim))
    if not claim_years or not re.search(r"\b(?:not|never)\b", normalized_claim):
        return False
    claim_terms = {
        token for token in re.findall(r"[a-z0-9]+", normalized_claim)
        if len(token) > 2 and token not in claim_years | {"not", "never"}
    }
    for source in evidence:
        entity_alignment = _entity_text_alignment(claim, f"{source.title} {source.snippet}")
        if entity_alignment < 0.67:
            continue
        for sentence in re.split(r"(?<=[.!?;])\s+", source.snippet):
            sentence_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", _normalize(sentence)))
            if not sentence_years or sentence_years & claim_years:
                continue
            sentence_terms = set(re.findall(r"[a-z0-9]+", _normalize(sentence)))
            overlap = len(claim_terms & sentence_terms) / len(claim_terms) if claim_terms else 0.0
            if overlap >= 0.55:
                return True
    return False


def _document_section_count_support(claim: str, evidence: list[EvidenceSource]) -> float:
    """Verify section counts from the structure of an uploaded document.

    PDF extraction often preserves headings but not a sentence literally saying
    "this syllabus contains ten sections". Counting explicit section labels is
    safer than asking an NLI model to infer a document's structure.
    """
    match = re.search(
        r"\b(?:syllabus|document|file)?\s*(?:contains|has|includes|comprises)\s+"
        r"(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+sections?\b",
        _normalize(claim),
    )
    if not match:
        return 0.0
    raw_count = match.group(1)
    expected = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS[raw_count]
    document_text = " ".join(
        _normalize(source.snippet)
        for source in evidence
        if source.url.startswith("document://")
    )
    if not document_text:
        return 0.0
    if re.search(
        rf"\b(?:contains|has|includes|comprises)\s+(?:{expected}|{raw_count})\s+sections?\b",
        document_text,
    ):
        return 0.94
    labels = {int(label) for label in re.findall(r"\bsection\s+(\d{1,2})\b", document_text)}
    return 0.90 if set(range(1, expected + 1)) <= labels else 0.0


def _has_conflicting_year_evidence(claim: str, evidence: list[EvidenceSource]) -> bool:
    """Catch a different year asserted for the same well-matched event.

    NLI can treat two nearly identical sentences as entailed even when one
    decisive year is changed.  This safeguard activates only when a sentence
    in the evidence has strong non-year term overlap with the claim but lists
    a different four-digit year.  It avoids penalising an article that merely
    contains several unrelated dates.
    """
    # An exact date match is stronger evidence than a differently formatted
    # year appearing elsewhere in the same biography.
    if (
        _has_direct_date_support(claim, evidence)
        or _has_direct_year_support(claim, evidence)
        or _has_negative_year_fact_support(claim, evidence)
    ):
        return False

    claim_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", _normalize(claim)))
    if not claim_years:
        return False

    claim_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(claim))
        if len(token) > 2 and token not in claim_years
    }
    if len(claim_terms) < 3:
        return False

    for source in evidence:
        entity_alignment = _entity_text_alignment(claim, f"{source.title} {source.snippet}")
        sentences = re.split(r"(?<=[.!?;])\s+", source.snippet)
        for sentence in sentences:
            sentence_years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", _normalize(sentence)))
            if not sentence_years or sentence_years & claim_years:
                continue
            sentence_terms = {
                token for token in re.findall(r"[a-z0-9]+", _normalize(sentence)) if len(token) > 2
            }
            overlap = len(claim_terms & sentence_terms) / len(claim_terms)
            # A different date must be attached to the same entity and event,
            # not merely appear on a loosely related timeline or list page.
            if entity_alignment >= 0.67 and overlap >= 0.55:
                return True
    return False


def _option_pair_support(claim: str, evidence: list[EvidenceSource]) -> float:
    """Recognise compact table cells such as ``CS: MA, ST, EC, IN``.

    PDF extraction commonly removes table columns, leaving a factual pair as
    adjacent codes rather than a grammatical sentence. This is used only after
    NLI finds neither entailment nor contradiction.
    """
    match = re.match(r"^(.+?) is an option for ([A-Za-z][A-Za-z0-9+#-]*)\.$", claim)
    if not match:
        return 0.0

    option, subject = match.groups()
    code_match = re.search(r"\(([A-Za-z0-9+#-]{2,})\)", option)
    option_terms = [code_match.group(1)] if code_match else re.findall(r"[A-Za-z0-9]{3,}", option)
    if not option_terms:
        return 0.0

    for source in evidence:
        text = _normalize(source.snippet)
        # A colon marks an unambiguous compact row (for example
        # ``CS: MA, ST, EC``). Restrict the lookup to that row; otherwise an
        # option from the following ``DA: ...`` row could be credited to CS.
        row_pattern = re.compile(
            rf"\b{re.escape(subject.lower())}\b\s*:\s*"
            r"(?P<options>.*?)(?=\s+\b[a-z]{2}\b\s*:|[.;\n]|$)"
        )
        for row_match in row_pattern.finditer(text):
            options_text = row_match.group("options")
            if all(re.search(rf"\b{re.escape(term.lower())}\b", options_text) for term in option_terms):
                return 0.88
    return 0.0


def _claim_evidence_excerpt(claim: str, source: EvidenceSource) -> str:
    """Give NLI a focused premise instead of a long search-result paragraph."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;])\s+", source.snippet)
        if len(sentence.strip()) >= 20
    ]
    if not sentences:
        return f"{source.title}. {source.snippet[:900]}"

    semantic_scores = _semantic_sentence_scores(claim, sentences)
    ranked = sorted(
        (
            (
                0.65 * _evidence_match(claim, EvidenceSource(title=source.title, url=source.url, snippet=sentence))
                + 0.35 * semantic_scores[index]
                if semantic_scores is not None
                else _evidence_match(claim, EvidenceSource(title=source.title, url=source.url, snippet=sentence)),
                sentence,
            )
            for index, sentence in enumerate(sentences)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [sentence for _, sentence in ranked[:2]]
    return f"{source.title}. {' '.join(selected)}"


def _claim_evidence(
    claim: str,
    evidence: list[EvidenceSource],
    *,
    limit: int = 2,
) -> list[EvidenceSource]:
    """Keep unrelated global sources out of an individual NLI decision."""
    ranked = _rank_claim_sources(claim, evidence, limit=limit, minimum_score=0.18)
    if ranked:
        return [source for _score, source in ranked]
    # A named factual claim should not fall back to an arbitrary global page:
    # no evidence is safer and more honest than a tangential contradiction.
    if _claim_subject(claim):
        return []
    fallback = sorted(
        ((_evidence_match(claim, source), source) for source in evidence),
        key=lambda item: item[0],
        reverse=True,
    )
    return [fallback[0][1]] if fallback else []


def _evidence_quality(evidence: list[EvidenceSource]) -> float:
    """Average quality of the independent sources used for one claim."""
    if not evidence:
        return 0.0
    return round(sum(source.credibility for source in evidence) / len(evidence), 2)


def _source_agreement(scores: list[dict[str, float]], evidence: list[EvidenceSource]) -> float:
    """Estimate corroboration from NLI verdicts and independent source domains."""
    if not evidence:
        return 0.0

    entailment_votes = sum(
        score["entailment"] >= 0.55 and score["entailment"] > score["contradiction"]
        for score in scores
    )
    contradiction_votes = sum(
        score["contradiction"] >= 0.55 and score["contradiction"] > score["entailment"]
        for score in scores
    )
    decisive_consensus = (
        max(entailment_votes, contradiction_votes) / len(scores)
        if entailment_votes or contradiction_votes
        else 0.5
    )
    independent_domains = len({_source_host(source) for source in evidence})
    diversity = min(1.0, 0.60 + 0.20 * max(0, independent_domains - 1))
    return round(0.70 * decisive_consensus + 0.30 * diversity, 2)


def select_claim_citations(
    claim: str,
    evidence: list[EvidenceSource],
    *,
    limit: int = 2,
) -> list[EvidenceSource]:
    """Return only high-match, independent sources for one displayed claim."""
    return [
        source
        for _score, source in _rank_claim_sources(
            claim,
            evidence,
            limit=limit,
            minimum_score=0.42,
        )
    ]


@lru_cache
def _nli_model():
    """Load once, on the first verified answer instead of during API startup."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(NLI_MODEL)
    except Exception as exc:  # Model download, package, or local-runtime failure.
        raise NLIUnavailable(
            "Local NLI verification is unavailable. Install the backend requirements and retry."
        ) from exc


def _score_labels(model, scores: list[float]) -> dict[str, float]:
    labels = [str(model.model.config.id2label.get(index, "")).lower() for index in range(len(scores))]
    mapped: dict[str, float] = {}
    for label, score in zip(labels, scores, strict=True):
        if "contradict" in label:
            mapped["contradiction"] = float(score)
        elif "entail" in label:
            mapped["entailment"] = float(score)
        elif "neutral" in label:
            mapped["neutral"] = float(score)

    # The official model uses this index order. Keep this fallback for older
    # Transformers versions that expose LABEL_0/LABEL_1/LABEL_2.
    if len(mapped) != 3 and len(scores) == 3:
        mapped = {
            "contradiction": float(scores[0]),
            "entailment": float(scores[1]),
            "neutral": float(scores[2]),
        }
    return mapped


def _nli_verdict(claim: str, evidence: list[EvidenceSource]) -> tuple[str, float, str, float]:
    if not evidence:
        return "unsupported", 0.0, "No evidence source was available for this claim.", 0.0

    document_section_support = _document_section_count_support(claim, evidence)
    if document_section_support:
        return (
            "supported",
            document_section_support,
            "The uploaded document's explicit section headings support this count.",
            0.88,
        )
    if _has_direct_date_support(claim, evidence):
        return (
            "supported",
            0.95,
            "Retrieved evidence directly matches the date and subject in this claim.",
            0.88,
        )
    if _has_negative_year_fact_support(claim, evidence):
        return (
            "supported",
            0.93,
            "Retrieved evidence gives a different year, which supports this negated claim.",
            0.88,
        )
    if _has_direct_year_support(claim, evidence):
        return (
            "supported",
            0.92,
            "Retrieved evidence directly matches the year, subject, and event in this claim.",
            0.88,
        )
    if _collapses_disputed_report_into_fact(claim, evidence):
        return (
            "unsupported",
            0.82,
            "Retrieved evidence describes this event as unverified or disputed, not an established fact.",
            0.55,
        )
    if _has_conflicting_year_evidence(claim, evidence):
        return (
            "unsupported",
            0.86,
            "Retrieved evidence gives a different year for the same event.",
            0.70,
        )

    model = _nli_model()
    pairs = [(_claim_evidence_excerpt(claim, source), claim) for source in evidence]
    predictions = model.predict(pairs, apply_softmax=True)
    scores = [_score_labels(model, [float(value) for value in row]) for row in predictions]

    best_entailment = max(score["entailment"] for score in scores)
    best_contradiction = max(score["contradiction"] for score in scores)
    best_neutral = max(score["neutral"] for score in scores)

    entailment_votes = sum(
        score["entailment"] >= 0.55 and score["entailment"] > score["contradiction"]
        for score in scores
    )
    contradiction_votes = sum(
        score["contradiction"] >= 0.55 and score["contradiction"] > score["entailment"]
        for score in scores
    )
    agreement = _source_agreement(scores, evidence)
    strong_textual_support = _strong_textual_support(claim, evidence)

    # A single mismatched source must not turn a factual answer into a false
    # hallucination. With multiple sources, require agreement before returning
    # the strongest (unsupported) verdict.
    if (
        best_contradiction >= 0.55
        and contradiction_votes > entailment_votes
        and (len(scores) == 1 or contradiction_votes >= 2)
    ):
        # NLI models can occasionally invert an obvious entailment, especially
        # for long biographies with names, offices, and dates. Override only a
        # single-source outlier when the evidence is a near-verbatim match and
        # the claim's numbers and negation polarity agree with that evidence.
        if strong_textual_support >= 0.82 and contradiction_votes == 1:
            confidence = min(0.92, 0.55 + 0.45 * strong_textual_support)
            return (
                "supported",
                confidence,
                "Retrieved evidence directly matches the factual content of this claim.",
                agreement,
            )
        direct_single_source = any(
            _entity_text_alignment(claim, f"{source.title} {source.snippet}") >= 0.90
            and _evidence_match(claim, source) >= 0.45
            for source in evidence
        )
        high_quality_direct_source = any(
            source.credibility >= 0.84 and _evidence_match(claim, source) >= 0.55
            for source in evidence
        )
        # A lone NLI contradiction is useful only when the cited source is
        # clearly about the same claim. Otherwise it remains a review item,
        # preventing weak or tangential pages from falsely labelling an answer
        # as hallucinated.
        if contradiction_votes == 1 and not (direct_single_source or high_quality_direct_source):
            return (
                "uncertain",
                best_contradiction,
                "The retrieved source is too indirect to establish a contradiction for this claim.",
                agreement,
            )
        return "unsupported", best_contradiction, "An NLI model found the claim contradicted by retrieved evidence.", agreement
    if (
        best_entailment >= 0.55
        and entailment_votes > contradiction_votes
        and best_entailment >= best_neutral
    ):
        return "supported", best_entailment, "An NLI model found the claim entailed by retrieved evidence.", agreement
    if entailment_votes and contradiction_votes:
        return "uncertain", max(best_entailment, best_contradiction), "Retrieved sources do not agree strongly enough to verify this claim.", agreement
    option_pair_support = _option_pair_support(claim, evidence)
    if option_pair_support and not contradiction_votes:
        return (
            "supported",
            option_pair_support,
            "The retrieved document table lists this option for the requested subject.",
            agreement,
        )
    lexical_support = strong_textual_support
    if lexical_support >= 0.68 and not contradiction_votes:
        confidence = min(0.92, 0.55 + 0.45 * lexical_support)
        return "supported", confidence, "Retrieved evidence closely matches the factual content of this claim.", agreement
    return "uncertain", best_neutral, "Retrieved evidence does not clearly entail or contradict this claim.", agreement


def _winner_list_claims(answer: str, question: str, *, limit: int = 6) -> list[str]:
    """Turn numbered winner lists into complete claims before NLI scoring.

    A raw item such as ``1. Nethra Raghuraman`` is not a factual statement by
    itself, so it used to be merged with the next number and falsely marked as
    contradicted. The question supplies the missing relationship.
    """
    if not re.search(r"\b(?:winner|winners|won)\b", question, flags=re.IGNORECASE):
        return []
    items = re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", answer.strip())
    if len(items) < 2:
        return []

    subject = re.sub(
        r"\b(?:all|list|name|the|of|winner|winners|won|till|to|date|season|seasons)\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    subject = re.sub(r"\s+", " ", subject).strip(" ?!.,")
    if not subject:
        return []

    claims: list[str] = []
    for item in items:
        name = re.sub(r"\s*\((?:season\s*)?\d+[^)]*\)", "", item, flags=re.IGNORECASE)
        name = name.strip(" .")
        if len(name) >= 3:
            claims.append(f"{name} was a winner of {subject}.")
    return claims[:limit]


def _factual_list_claims(answer: str, question: str, *, limit: int = 30) -> list[str]:
    """Make numbered factual lists verifiable without treating item numbers as sentences."""
    if not re.search(r"\b(?:list|name|papers?|options?|items?)\b", question, flags=re.IGNORECASE):
        return []
    items = re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", answer.strip())
    if len(items) < 2:
        return []

    header = answer[: re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", answer).start()]
    context_match = re.search(r"\b(?:GATE\s*\d{4}\s+)?test papers?\b", f"{question} {header}", re.IGNORECASE)
    if not context_match:
        return []
    context = context_match.group(0)
    if not re.search(r"GATE\s*\d{4}", context, re.IGNORECASE):
        gate_match = re.search(r"GATE\s*\d{4}", f"{question} {header}", re.IGNORECASE)
        if gate_match:
            context = f"{gate_match.group(0)} {context}"

    claims = []
    for item in items:
        cleaned = item.strip().rstrip(".")
        if len(cleaned) >= 2:
            claims.append(f"{cleaned} is included in the list of {context}.")
    return claims[:limit]


def _grouped_option_list_claims(answer: str, question: str, *, limit: int = 30) -> list[str]:
    """Turn grouped option lists into complete claims for each named subject.

    For example, a response with separate numbered options for CS and DA must
    not be split into fragments such as ``Electronics ... 4.``.
    """
    if not re.search(r"\b(?:options?|combinations?)\b", question, flags=re.IGNORECASE):
        return []

    claims: list[str] = []
    current_subject: str | None = None
    for line in answer.splitlines():
        header = re.match(
            r"^\s*(?:for\s+)?([A-Za-z][A-Za-z0-9+# -]{0,50}),?\s+"
            r"(?:the\s+)?(?:second\s+)?(?:paper\s+)?options?\s+(?:are|include)\s*:?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if header:
            current_subject = header.group(1).strip()
            continue

        item = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", line)
        if item and current_subject:
            option = item.group(1).strip().rstrip(".")
            if len(option) >= 2:
                claims.append(f"{option} is an option for {current_subject}.")

    return claims[:limit] if len(claims) >= 2 else []


def _is_conversational_sentence(sentence: str) -> bool:
    """Identify responses that do not make an externally checkable claim.

    A verifier should not label a greeting, acknowledgement, question, or a
    personal preference as hallucinated merely because a web source cannot
    entail it. This deliberately recognises only clear conversational forms;
    factual sentences in a mixed response still proceed to NLI verification.
    """
    normalized = _normalize(sentence).strip(" .!?")
    if not normalized or sentence.strip().endswith("?"):
        return True

    exact_phrases = {
        "yes", "no", "okay", "ok", "sure", "of course", "certainly",
        "absolutely", "thanks", "thank you", "you're welcome", "youre welcome", "your welcome",
        "no problem", "my pleasure", "i see", "sounds good",
    }
    if normalized in exact_phrases:
        return True

    patterns = (
        r"^(?:any(?:thing| other).{0,80}(?:help|know|question|ask).*)$",
        r"^(?:please )?(?:specify|clarify|tell me)\b.*$",
        r"^(?:i(?: am|'m) sorry|sorry)\b.*$",
        r"^i (?:do not|dont|don't) (?:know|have) (?:that |the )?(?:information|detail|context).*$",
        r"^i (?:like|love|prefer|enjoy|would say|believe|think)\b.*$",
        r"^(?:that|this|it) (?:is|was) (?:great|interesting|fun|nice|wonderful|helpful)\b.*$",
    )
    return any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _is_unresolved_reference_claim(claim: str) -> bool:
    """Skip vague follow-on sentences that cannot be independently checked.

    A sentence such as "The technology was developed at a company" loses the
    actual product name. Searching it can therefore cite any page containing
    the generic words "technology" and "company". The generator is also told
    to repeat the entity, but this guard keeps one vague sentence from
    contaminating a whole answer's reliability score.
    """
    generic_subjects = {
        "the technology", "the company", "the organization", "the organisation",
        "the project", "the language", "the institution", "the team", "the mission",
        "this technology", "this company", "this project", "this language",
        "it", "they", "this", "that",
    }
    return _normalize(_claim_subject(claim)) in generic_subjects


def _is_direct_answer_fragment(answer: str, question: str) -> bool:
    """Allow a short factual answer when its question supplies the relation.

    Answers in ordinary chat are often complete sentences.  In question-answer
    datasets, and sometimes in the product, the correct response can instead
    be a name, date, place, or title such as ``Michael Dowse``.  Treating every
    short reply as a factual claim would make acknowledgements such as ``yes``
    and ``okay`` look verifiable, so this accepts only a compact non-
    conversational fragment accompanied by a meaningful question.
    """
    compact = answer.strip().strip(" .")
    if not question or len(question.strip()) < 8 or not compact:
        return False
    if len(compact) < 2 or len(compact) > 90 or "\n" in compact or compact.endswith("?"):
        return False
    if _is_conversational_sentence(compact):
        return False
    return not re.search(r"\b(?:i|you|we)\s+(?:am|are|was|were|do|did|have|has)\b", compact, re.IGNORECASE)


def extract_claims(answer: str, question: str = "", *, limit: int = 6) -> list[str]:
    # Protect initialisms and titles before splitting sentences. Without this,
    # "earned an M.F.A. from Harvard" was incorrectly treated as two claims.
    winner_claims = _winner_list_claims(answer, question, limit=limit)
    if winner_claims:
        return winner_claims
    # Unlike ordinary prose, an explicit enumerated list can legitimately have
    # many entries. Preserve up to 30 for verification rather than silently
    # scoring only its first six items.
    factual_list_claims = _factual_list_claims(answer, question, limit=max(limit, 30))
    if factual_list_claims:
        return factual_list_claims
    grouped_option_claims = _grouped_option_list_claims(answer, question, limit=max(limit, 30))
    if grouped_option_claims:
        return grouped_option_claims

    marker = "<period>"

    def protect(match: re.Match[str]) -> str:
        return match.group(0).replace(".", marker)

    protected_answer = re.sub(r"\b(?:[A-Za-z]\.\s*){2,}", protect, answer.strip())
    protected_answer = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.",
        protect,
        protected_answer,
        flags=re.IGNORECASE,
    )
    sentences = [sentence.replace(marker, ".") for sentence in re.split(r"(?<=[.!?])\s+", protected_answer)]
    boilerplate = re.compile(r"^(?:based on (?:the )?(?:provided|retrieved) evidence,?\s*)", flags=re.IGNORECASE)
    insufficient_evidence = re.compile(
        r"^(?:the )?(?:supplied|provided|retrieved) (?:information|evidence) "
        r"(?:does not|cannot|is insufficient|is not sufficient)",
        flags=re.IGNORECASE,
    )
    claims: list[str] = []
    last_person: str | None = None
    person_subject = re.compile(
        r"^([A-Z][A-Za-z.'-]*(?:\s+(?:[A-Z][A-Za-z.'-]*|van|von|de|da|del)){1,4})\s+"
        r"(?:is|was|created|developed|released|designed|founded|led|served|became|worked|"
        r"shared|received|coined|isolated|organized|organised|won|died|passed)\b"
    )
    for sentence in sentences:
        cleaned = boilerplate.sub("", sentence).strip()
        if insufficient_evidence.match(cleaned):
            continue
        # Split only explicit clause boundaries. This complements the LLM prompt
        # without breaking natural lists such as "astronaut, engineer, and pilot".
        clauses = re.split(r"\s*;\s*|,?\s+and\s+(?=(?:he|she|they|it)\b)", cleaned, flags=re.IGNORECASE)
        for clause in clauses:
            claim = clause.strip()
            if _is_conversational_sentence(claim):
                continue
            if last_person and re.match(r"^(?:he|she)\b", claim, flags=re.IGNORECASE):
                claim = re.sub(r"^(?:he|she)\b", last_person, claim, flags=re.IGNORECASE)
            if _is_unresolved_reference_claim(claim):
                continue
            if len(claim) < 20:
                continue
            match = person_subject.match(claim)
            if match:
                last_person = match.group(1)
            claims.append(claim)
    # A direct answer like "Michael Dowse" has no verb of its own, but the
    # associated question gives it a checkable relationship.  Preserve it for
    # evidence matching only when no normal factual claim was already found.
    if not claims and _is_direct_answer_fragment(answer, question):
        claims.append(answer.strip().strip(" ."))
    return claims[:limit]


def limit_factual_answer(answer: str, *, question: str = "", limit: int = 6) -> str:
    """Ensure every displayed factual claim fits in one verification pass."""
    # Lists are formatted one item per line and may legitimately contain more
    # than six entries (for example a document's complete set of exam papers).
    if (
        _factual_list_claims(answer, question, limit=30)
        or _grouped_option_list_claims(answer, question, limit=30)
    ):
        return answer.strip()
    claims = extract_claims(answer, question, limit=limit + 1)
    if len(claims) <= limit:
        return answer.strip()
    return " ".join(claims[:limit])


def _fallback_assessment(claim: str, evidence: list[EvidenceSource], reason: str) -> ClaimAssessment:
    overlap = max((_evidence_match(claim, source) for source in evidence), default=0.0)
    status = "supported" if overlap >= 0.70 else "uncertain" if overlap >= 0.25 else "unsupported"
    return ClaimAssessment(
        claim=claim,
        status=status,
        confidence=round(overlap, 2),
        rationale=f"{reason} Keyword evidence match was used as a fallback.",
        evidence_quality=_evidence_quality(evidence),
        source_agreement=round(0.60 + 0.20 * max(0, len({_source_host(source) for source in evidence}) - 1), 2) if evidence else 0.0,
    )


def verify_claims(
    answer: str,
    evidence: list[EvidenceSource],
    *,
    question: str = "",
) -> list[ClaimAssessment]:
    claims = extract_claims(answer, question)
    if not claims:
        return []
    try:
        assessments: list[ClaimAssessment] = []
        for claim in claims:
            focused_evidence = _claim_evidence(claim, evidence)
            # Prefer the excerpt containing the compact subject→option table
            # pair when ordinary text ranking selected a neighbouring table row.
            option_table_evidence = [
                source for source in evidence if _option_pair_support(claim, [source])
            ]
            if option_table_evidence:
                focused_evidence = option_table_evidence[:2]
            status, confidence, rationale, agreement = _nli_verdict(claim, focused_evidence)
            citations = select_claim_citations(claim, focused_evidence)
            assessments.append(
                ClaimAssessment(
                    claim=claim,
                    status=status,
                    confidence=round(confidence, 2),
                    rationale=rationale,
                    citations=citations,
                    evidence_quality=_evidence_quality(citations or focused_evidence),
                    source_agreement=agreement,
                )
            )
        return assessments
    except NLIUnavailable as exc:
        return [
            _fallback_assessment(claim, _claim_evidence(claim, evidence), str(exc)).model_copy(
                update={
                    "citations": select_claim_citations(
                        claim,
                        _claim_evidence(claim, evidence),
                    )
                }
            )
            for claim in claims
        ]


def select_verification_sources(
    claims: list[ClaimAssessment],
    evidence: list[EvidenceSource],
    *,
    limit: int = 4,
) -> list[EvidenceSource]:
    """Show only evidence that was materially relevant to at least one claim."""
    best_by_url: dict[str, tuple[float, EvidenceSource]] = {}
    for assessment in claims:
        claim_sources = assessment.citations or select_claim_citations(assessment.claim, evidence)
        for source in claim_sources:
            score = _evidence_match(assessment.claim, source) * (0.85 + 0.15 * source.credibility)
            previous = best_by_url.get(source.url)
            if previous is None or score > previous[0]:
                best_by_url[source.url] = (score, source)
    ranked = sorted(best_by_url.values(), key=lambda item: item[0], reverse=True)
    return [source for score, source in ranked[:limit] if score >= 0.18]


def select_citations(answer: str, evidence: list[EvidenceSource], *, limit: int = 2) -> list[EvidenceSource]:
    """Return sources that substantively support an individual corrected claim.

    Ranking a whole correction can mistakenly cite a page that merely shares a
    country name. Claim-by-claim selection keeps citations relevant when an
    answer contains more than one factual statement.
    """
    best_by_url: dict[str, tuple[float, int, EvidenceSource]] = {}
    for claim_index, claim in enumerate(extract_claims(answer)):
        for source in select_claim_citations(claim, evidence):
            score = _evidence_match(claim, source) * (0.85 + 0.15 * source.credibility)
            previous = best_by_url.get(source.url)
            if previous is None or score > previous[0]:
                # Keep the first claim position even if this source supports a
                # later claim more strongly. The citation list is displayed
                # next to a corrected answer, so statement order is clearer
                # and more stable than a global score-only ordering.
                first_claim_index = previous[1] if previous else claim_index
                best_by_url[source.url] = (score, first_claim_index, source)
    ordered = sorted(best_by_url.values(), key=lambda item: item[1])
    return [source for score, _, source in ordered[:limit] if score >= 0.42]


def reliability_score(claims: list[ClaimAssessment]) -> float:
    if not claims:
        return 0.0
    weights = {"supported": 1.0, "uncertain": 0.5, "unsupported": 0.0}
    total = 0.0
    for claim in claims:
        # The verdict remains dominant. Source quality and agreement make a
        # small, transparent adjustment without allowing a good source to
        # turn an unsupported claim into a reliable one.
        quality = claim.evidence_quality if claim.evidence_quality is not None else 1.0
        agreement = claim.source_agreement if claim.source_agreement is not None else 1.0
        evidence_factor = 0.85 + 0.10 * quality + 0.05 * agreement
        total += weights[claim.status] * claim.confidence * evidence_factor
    return round(total / len(claims), 2)
