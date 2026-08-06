from app.services.verifier import (
    extract_claims,
    limit_factual_answer,
    reliability_score,
    select_claim_citations,
    select_citations,
    select_verification_sources,
    verify_claims,
)
from app.schemas import ClaimAssessment, EvidenceSource


def test_extract_claims_splits_sentences() -> None:
    answer = "Python was created by Guido van Rossum. It first appeared in 1991."
    claims = extract_claims(answer)
    assert len(claims) == 2


def test_factual_answer_is_trimmed_to_the_verifiable_claim_limit() -> None:
    answer = " ".join(
        [
            "The first factual statement contains enough detail to verify.",
            "The second factual statement contains enough detail to verify.",
            "The third factual statement contains enough detail to verify.",
            "The fourth factual statement contains enough detail to verify.",
            "The fifth factual statement contains enough detail to verify.",
            "The sixth factual statement contains enough detail to verify.",
            "The seventh factual statement must not be displayed unverified.",
        ]
    )

    limited = limit_factual_answer(answer)

    assert "seventh factual statement" not in limited
    assert limited.count("statement contains enough detail") == 6


def test_extract_claims_splits_explicit_pronoun_clause() -> None:
    answer = "Bjarne Stroustrup created C++, and he developed it at Bell Labs."
    claims = extract_claims(answer)
    assert claims == [
        "Bjarne Stroustrup created C++",
        "Bjarne Stroustrup developed it at Bell Labs.",
    ]


def test_extract_claims_keeps_initialisms_in_one_claim() -> None:
    claims = extract_claims(
        "Jon Bernthal earned an M.F.A. from Harvard University's Institute for Advanced Theatre Training."
    )
    assert claims == [
        "Jon Bernthal earned an M.F.A. from Harvard University's Institute for Advanced Theatre Training."
    ]


def test_extract_claims_ignores_evidence_refusal_sentence() -> None:
    answer = (
        "C. K. Nayudu was the first Test captain of India. "
        "The supplied information does not name all the players in that team."
    )
    assert extract_claims(answer) == ["C. K. Nayudu was the first Test captain of India."]


def test_extract_claims_ignores_conversational_only_reply() -> None:
    assert extract_claims("You're welcome. Anything else I can help with?") == []


def test_extract_claims_keeps_fact_after_conversational_sentence() -> None:
    assert extract_claims("Of course. Guido van Rossum created Python.") == [
        "Guido van Rossum created Python."
    ]


def test_numbered_winner_list_becomes_complete_claims() -> None:
    answer = "1. Nethra Raghuraman\n2. Anushka Manchanda\n3. Shabir Ahluwalia"

    assert extract_claims(answer, "Khatron Ke Khiladi winners till date") == [
        "Nethra Raghuraman was a winner of Khatron Ke Khiladi.",
        "Anushka Manchanda was a winner of Khatron Ke Khiladi.",
        "Shabir Ahluwalia was a winner of Khatron Ke Khiladi.",
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


def test_citations_are_selected_per_corrected_claim() -> None:
    correction = "P. V. Narasimha Rao was the 9th Prime Minister of India. Ulysses S. Grant was the 18th President of the United States."
    india = EvidenceSource(
        title="P. V. Narasimha Rao",
        url="https://example.com/rao",
        snippet="P. V. Narasimha Rao served as the ninth prime minister of India.",
    )
    united_states = EvidenceSource(
        title="Ulysses S. Grant",
        url="https://example.com/grant",
        snippet="Ulysses S. Grant was the 18th president of the United States.",
    )
    unrelated = EvidenceSource(
        title="India-Indonesia relations",
        url="https://example.com/relations",
        snippet="India and Indonesia have longstanding diplomatic relations.",
    )

    citations = select_citations(correction, [unrelated, india, united_states])
    assert citations == [india, united_states]


def test_claim_citations_require_a_strong_match_and_distinct_domains() -> None:
    claim = "Guido van Rossum created Python."
    relevant = EvidenceSource(
        title="Python",
        url="https://www.python.org/doc/essays/blurb/",
        snippet="Python was created by Guido van Rossum and first released in 1991.",
        credibility=0.95,
    )
    duplicate_domain = EvidenceSource(
        title="Python history",
        url="https://www.python.org/history/",
        snippet="Guido van Rossum created Python in the late 1980s.",
        credibility=0.95,
    )
    unrelated = EvidenceSource(
        title="JavaScript",
        url="https://example.com/javascript",
        snippet="JavaScript was created by Brendan Eich.",
    )

    citations = select_claim_citations(claim, [unrelated, duplicate_domain, relevant])
    assert len(citations) == 1
    assert citations[0].url.startswith("https://www.python.org/")


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


def test_reliability_rewards_credible_and_independent_evidence() -> None:
    high_quality = ClaimAssessment(
        claim="Python was created by Guido van Rossum.",
        status="supported",
        confidence=0.90,
        rationale="Supported.",
        evidence_quality=0.95,
        source_agreement=1.0,
    )
    lower_quality = ClaimAssessment(
        claim="Python was created by Guido van Rossum.",
        status="supported",
        confidence=0.90,
        rationale="Supported.",
        evidence_quality=0.35,
        source_agreement=0.60,
    )

    assert reliability_score([high_quality]) > reliability_score([lower_quality])


def test_unsupported_claim_never_receives_reliability_from_source_quality() -> None:
    unsupported = ClaimAssessment(
        claim="Python was created by someone else.",
        status="unsupported",
        confidence=1.0,
        rationale="Contradicted.",
        evidence_quality=0.95,
        source_agreement=1.0,
    )

    assert reliability_score([unsupported]) == 0.0
