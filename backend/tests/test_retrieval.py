from app.schemas import EvidenceSource
from app.services.retrieval import (
    _split_compound_question,
    _has_topic_anchor,
    _is_unrelated_identity_page,
    _keywords,
    _relation_parts,
    _research_query,
    _select_diverse_sources,
    _source_merge_key,
    _subject_query,
)


def test_creator_question_preserves_symbolic_subject() -> None:
    question = "Who is the creator of C++?"

    assert _relation_parts(question) == ("C++", "creator")
    assert _research_query(question) == '"C++" creator'
    assert "cplusplus" in _keywords(question)


def test_creator_question_rejects_unrelated_creator_result() -> None:
    question = "Who is the creator of C++?"
    unrelated = EvidenceSource(
        title="Tyler, the Creator",
        url="https://example.com/tyler",
        snippet="Tyler, the Creator is an American musician.",
    )
    relevant = EvidenceSource(
        title="Bjarne Stroustrup",
        url="https://example.com/stroustrup",
        snippet="Bjarne Stroustrup created and developed the C++ programming language.",
    )

    assert not _has_topic_anchor(question, unrelated)
    assert _has_topic_anchor(question, relevant)


def test_common_technical_aliases_are_canonicalized() -> None:
    assert _relation_parts("Who created JS?") == ("JavaScript", "creator")
    assert _subject_query("Who is JS?") == "JavaScript"


def test_compound_ordinal_question_is_split_for_retrieval() -> None:
    assert _split_compound_question(
        "Who was the 9th PM of India and 18th President of USA?"
    ) == [
        "Who was the 9th PM of India?",
        "Who was the 18th President of USA?",
    ]


def test_identity_question_with_achievement_request_keeps_person_subject() -> None:
    assert _subject_query("Who is Prakash Padukone and list his achievements?") == "Prakash Padukone"
    assert _research_query("Who is Prakash Padukone and list his achievements?") == "Prakash Padukone achievements career"


def test_biographical_follow_up_uses_the_person_not_a_namesake_institution() -> None:
    assert _subject_query("When was Kalpana Chawla born?") == "Kalpana Chawla"
    assert _research_query("When was Kalpana Chawla born?") == "Kalpana Chawla birth date birthplace"


def test_identity_search_rejects_a_different_person_with_the_same_name() -> None:
    namesake = EvidenceSource(
        title="Pilli Subhash Chandra Bose",
        url="https://example.com/pilli-bose",
        snippet="Pilli Subhash Chandra Bose is an Indian politician.",
    )
    profile = EvidenceSource(
        title="Dr. Kalpana Chawla | National Air and Space Museum",
        url="https://airandspace.si.edu/kalpana-chawla",
        snippet="Kalpana Chawla was an astronaut.",
    )

    assert _is_unrelated_identity_page("Who is Subhash Chandra Bose?", namesake)
    assert not _is_unrelated_identity_page("Who is Kalpana Chawla?", profile)


def test_diverse_evidence_does_not_count_one_domain_multiple_times() -> None:
    wikipedia_profile = EvidenceSource(
        title="Python",
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        snippet="Python is a programming language created by Guido van Rossum.",
    )
    wikipedia_history = EvidenceSource(
        title="History of Python",
        url="https://en.wikipedia.org/wiki/History_of_Python",
        snippet="Python was first released in 1991.",
    )
    official = EvidenceSource(
        title="Python history",
        url="https://www.python.org/doc/essays/blurb/",
        snippet="Python was created by Guido van Rossum.",
    )

    selected = _select_diverse_sources(
        [(wikipedia_profile, 10.0), (wikipedia_history, 9.0), (official, 8.0)],
        limit=3,
    )
    assert selected == [wikipedia_profile, official]


def test_same_title_from_independent_domains_is_not_merged() -> None:
    official = EvidenceSource(
        title="Microsoft",
        url="https://www.microsoft.com/en-us/about",
        snippet="Microsoft is a technology company.",
    )
    encyclopedia = EvidenceSource(
        title="Microsoft",
        url="https://en.wikipedia.org/wiki/Microsoft",
        snippet="Microsoft is an American technology corporation.",
    )

    assert _source_merge_key(official) != _source_merge_key(encyclopedia)
