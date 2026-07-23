from app.services.verifier import extract_claims, reliability_score, verify_claims
from app.schemas import EvidenceSource


def test_extract_claims_splits_sentences() -> None:
    answer = "Python was created by Guido van Rossum. It first appeared in 1991."
    claims = extract_claims(answer)
    assert len(claims) == 2


def test_verify_claims_marks_supported_overlap() -> None:
    evidence = [
        EvidenceSource(
            title="Python",
            url="https://example.com/python",
            snippet="Python was created by Guido van Rossum and first released in 1991.",
        )
    ]
    answer = "Python was created by Guido van Rossum and first released in 1991."
    claims = verify_claims(answer, evidence)
    assert claims[0].status == "supported"
    assert reliability_score(claims) > 0.5
