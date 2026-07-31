from __future__ import annotations

from functools import lru_cache
import re

from app.schemas import ClaimAssessment, EvidenceSource

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class NLIUnavailable(RuntimeError):
    """Raised when the optional local NLI model cannot be loaded."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


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
    pairs = [(source.snippet, claim) for source in evidence]
    predictions = model.predict(pairs, apply_softmax=True)
    scores = [_score_labels(model, [float(value) for value in row]) for row in predictions]

    best_entailment = max(score["entailment"] for score in scores)
    best_contradiction = max(score["contradiction"] for score in scores)
    best_neutral = max(score["neutral"] for score in scores)

    if best_contradiction >= 0.55 and best_contradiction > best_entailment:
        return "unsupported", best_contradiction, "An NLI model found the claim contradicted by retrieved evidence."
    if best_entailment >= 0.55 and best_entailment >= best_neutral:
        return "supported", best_entailment, "An NLI model found the claim entailed by retrieved evidence."
    return "uncertain", best_neutral, "Retrieved evidence does not clearly entail or contradict this claim."


def extract_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    boilerplate = re.compile(r"^(?:based on (?:the )?(?:provided|retrieved) evidence,?\s*)", flags=re.IGNORECASE)
    claims = [boilerplate.sub("", sentence).strip() for sentence in sentences if len(sentence.strip()) >= 20]
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
        return [
            ClaimAssessment(claim=claim, status=status, confidence=round(confidence, 2), rationale=rationale)
            for claim in claims
            for status, confidence, rationale in [_nli_verdict(claim, evidence)]
        ]
    except NLIUnavailable as exc:
        return [_fallback_assessment(claim, evidence, str(exc)) for claim in claims]


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
