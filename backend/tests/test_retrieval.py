from app.schemas import EvidenceSource
from app.services.retrieval import (
    _has_topic_anchor,
    _keywords,
    _relation_parts,
    _research_query,
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
