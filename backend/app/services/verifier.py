from __future__ import annotations

import re

from app.schemas import ClaimAssessment, EvidenceSource


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[a-z0-9]+", _normalize(left)) if len(token) > 2}
    right_tokens = {token for token in re.findall(r"[a-z0-9]+", _normalize(right)) if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _evidence_match(claim: str, source: EvidenceSource) -> float:
    """Favor evidence that covers the factual terms of the individual claim."""
    claim_tokens = {
        token for token in re.findall(r"[a-z0-9]+", _normalize(claim)) if len(token) > 2
    }
    evidence_text = f"{source.title} {source.snippet}"
    evidence_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(evidence_text))
        if len(token) > 2
    }
    if not claim_tokens or not evidence_tokens:
        return 0.0

    coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    similarity = _token_overlap(claim, evidence_text)
    return 0.8 * coverage + 0.2 * similarity


def extract_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    boilerplate = re.compile(
        r"^(?:based on (?:the )?(?:provided|retrieved) evidence,?\s*)",
        flags=re.IGNORECASE,
    )
    claims = [
        boilerplate.sub("", sentence).strip()
        for sentence in sentences
        if len(sentence.strip()) >= 20
    ]
    return claims[:6] or [answer.strip()]


def verify_claims(answer: str, evidence: list[EvidenceSource]) -> list[ClaimAssessment]:
    assessments: list[ClaimAssessment] = []

    for claim in extract_claims(answer):
        overlap = max((_evidence_match(claim, source) for source in evidence), default=0.0)
        if overlap >= 0.70:
            status = "supported"
            rationale = "Claim is strongly covered by a retrieved evidence source."
        elif overlap >= 0.40:
            status = "uncertain"
            rationale = "Claim is partially reflected in retrieved evidence."
        else:
            status = "unsupported"
            rationale = "Claim is not sufficiently covered by the retrieved evidence."

        assessments.append(
            ClaimAssessment(
                claim=claim,
                status=status,
                confidence=round(overlap, 2),
                rationale=rationale,
            )
        )

    return assessments


def reliability_score(claims: list[ClaimAssessment]) -> float:
    if not claims:
        return 0.0

    weights = {"supported": 1.0, "uncertain": 0.5, "unsupported": 0.0}
    total = sum(weights[claim.status] * claim.confidence for claim in claims)
    return round(total / len(claims), 2)
