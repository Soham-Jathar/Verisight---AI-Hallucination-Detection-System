from app.services.verifier import (
    extract_claims,
    reliability_score,
    select_verification_sources,
    verify_claims,
)
from app.schemas import ClaimAssessment, EvidenceSource


def test_extract_claims_splits_sentences() -> None:
    answer = "Python was created by Guido van Rossum. It first appeared in 1991."
    claims = extract_claims(answer)
    assert len(claims) == 2


def test_extract_claims_splits_explicit_pronoun_clause() -> None:
    answer = "Bjarne Stroustrup created C++, and he developed it at Bell Labs."
    claims = extract_claims(answer)
    assert claims == [
        "Bjarne Stroustrup created C++",
        "Bjarne Stroustrup developed it at Bell Labs.",
    ]


def test_verification_sources_exclude_unrelated_pages() -> None:
    claim = ClaimAssessment(
        claim="Bjarne Stroustrup created C++.",
        status="supported",
        confidence=0.95,
        rationale="Supported by evidence.",
    )
    relevant = EvidenceSource(
        title="Bjarne Stroustrup",
        url="https://example.com/cpp",
        snippet="Bjarne Stroustrup created the C++ programming language.",
    )
    unrelated = EvidenceSource(
        title="Tyler, the Creator",
        url="https://example.com/music",
        snippet="An American musician and producer.",
    )

    assert select_verification_sources([claim], [unrelated, relevant]) == [relevant]


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
