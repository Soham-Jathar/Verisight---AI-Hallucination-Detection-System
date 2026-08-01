from __future__ import annotations

from functools import lru_cache
import re
import unicodedata

from app.schemas import ClaimAssessment, EvidenceSource

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class NLIUnavailable(RuntimeError):
    """Raised when the optional local NLI model cannot be loaded."""


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


def _claim_evidence_excerpt(claim: str, source: EvidenceSource) -> str:
    """Give NLI a focused premise instead of a long search-result paragraph."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;])\s+", source.snippet)
        if len(sentence.strip()) >= 20
    ]
    if not sentences:
        return f"{source.title}. {source.snippet[:900]}"

    ranked = sorted(
        (
            (_evidence_match(claim, EvidenceSource(title=source.title, url=source.url, snippet=sentence)), sentence)
            for sentence in sentences
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
    ranked = sorted(
        ((_evidence_match(claim, source), source) for source in evidence),
        key=lambda item: item[0],
        reverse=True,
    )
    focused = [source for score, source in ranked if score >= 0.18]
    return focused[:limit] or ([ranked[0][1]] if ranked else [])


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


def _nli_verdict(claim: str, evidence: list[EvidenceSource]) -> tuple[str, float, str]:
    if not evidence:
        return "unsupported", 0.0, "No evidence source was available for this claim."

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

    # A single mismatched source must not turn a factual answer into a false
    # hallucination. With multiple sources, require agreement before returning
    # the strongest (unsupported) verdict.
    if (
        best_contradiction >= 0.55
        and contradiction_votes > entailment_votes
        and (len(scores) == 1 or contradiction_votes >= 2)
    ):
        return "unsupported", best_contradiction, "An NLI model found the claim contradicted by retrieved evidence."
    if (
        best_entailment >= 0.55
        and entailment_votes > contradiction_votes
        and best_entailment >= best_neutral
    ):
        return "supported", best_entailment, "An NLI model found the claim entailed by retrieved evidence."
    if entailment_votes and contradiction_votes:
        return "uncertain", max(best_entailment, best_contradiction), "Retrieved sources do not agree strongly enough to verify this claim."
    lexical_support = _lexical_support(claim, evidence)
    if lexical_support >= 0.68 and not contradiction_votes:
        confidence = min(0.92, 0.55 + 0.45 * lexical_support)
        return "supported", confidence, "Retrieved evidence closely matches the factual content of this claim."
    return "uncertain", best_neutral, "Retrieved evidence does not clearly entail or contradict this claim."


def extract_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    boilerplate = re.compile(r"^(?:based on (?:the )?(?:provided|retrieved) evidence,?\s*)", flags=re.IGNORECASE)
    claims: list[str] = []
    last_person: str | None = None
    person_subject = re.compile(
        r"^([A-Z][A-Za-z.'-]*(?:\s+(?:[A-Z][A-Za-z.'-]*|van|von|de|da|del)){1,4})\s+"
        r"(?:is|was|created|developed|released|designed|founded|led|served|became|worked)\b"
    )
    for sentence in sentences:
        cleaned = boilerplate.sub("", sentence).strip()
        # Split only explicit clause boundaries. This complements the LLM prompt
        # without breaking natural lists such as "astronaut, engineer, and pilot".
        clauses = re.split(r"\s*;\s*|,?\s+and\s+(?=(?:he|she|they|it)\b)", cleaned, flags=re.IGNORECASE)
        for clause in clauses:
            claim = clause.strip()
            if last_person and re.match(r"^(?:he|she)\b", claim, flags=re.IGNORECASE):
                claim = re.sub(r"^(?:he|she)\b", last_person, claim, flags=re.IGNORECASE)
            if len(claim) < 20:
                continue
            match = person_subject.match(claim)
            if match:
                last_person = match.group(1)
            claims.append(claim)
    return claims[:6] or [answer.strip()]


def _fallback_assessment(claim: str, evidence: list[EvidenceSource], reason: str) -> ClaimAssessment:
    overlap = max((_evidence_match(claim, source) for source in evidence), default=0.0)
    status = "supported" if overlap >= 0.70 else "uncertain" if overlap >= 0.25 else "unsupported"
    return ClaimAssessment(
        claim=claim,
        status=status,
        confidence=round(overlap, 2),
        rationale=f"{reason} Keyword evidence match was used as a fallback.",
    )


def verify_claims(answer: str, evidence: list[EvidenceSource]) -> list[ClaimAssessment]:
    claims = extract_claims(answer)
    try:
        assessments: list[ClaimAssessment] = []
        for claim in claims:
            focused_evidence = _claim_evidence(claim, evidence)
            status, confidence, rationale = _nli_verdict(claim, focused_evidence)
            assessments.append(
                ClaimAssessment(
                    claim=claim,
                    status=status,
                    confidence=round(confidence, 2),
                    rationale=rationale,
                )
            )
        return assessments
    except NLIUnavailable as exc:
        return [
            _fallback_assessment(claim, _claim_evidence(claim, evidence), str(exc))
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
        for source in _claim_evidence(assessment.claim, evidence):
            score = _evidence_match(assessment.claim, source) * (0.85 + 0.15 * source.credibility)
            previous = best_by_url.get(source.url)
            if previous is None or score > previous[0]:
                best_by_url[source.url] = (score, source)
    ranked = sorted(best_by_url.values(), key=lambda item: item[0], reverse=True)
    return [source for score, source in ranked[:limit] if score >= 0.18]


def select_citations(answer: str, evidence: list[EvidenceSource], *, limit: int = 2) -> list[EvidenceSource]:
    """Return only sources whose title/excerpt materially matches the correction."""
    ranked = sorted(((_evidence_match(answer, source), source) for source in evidence), key=lambda item: item[0], reverse=True)
    return [source for score, source in ranked[:limit] if score >= 0.30]


def reliability_score(claims: list[ClaimAssessment]) -> float:
    if not claims:
        return 0.0
    weights = {"supported": 1.0, "uncertain": 0.5, "unsupported": 0.0}
    total = sum(weights[claim.status] * claim.confidence for claim in claims)
    return round(total / len(claims), 2)
