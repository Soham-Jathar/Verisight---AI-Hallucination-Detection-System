import pytest

from app.services.math_verifier import verify_math_answer
from app.services.question_types import (
    RequestKind,
    is_math_question,
    is_recommendation_request,
    normalize_math_shorthand,
    resolve_contextual_question,
    resolve_math_follow_up,
    route_request,
)
from app.schemas import ChatMessage


def test_gift_ideas_are_not_fact_checked() -> None:
    assert is_recommendation_request("List 5 gift ideas for professor")
    assert is_recommendation_request("Give 5 sites related to calculus formulas")
    assert is_recommendation_request("5 gift to give to friend")
    assert is_recommendation_request("9 must read books")
    assert is_recommendation_request("5 financial books")
    assert is_recommendation_request("5 selfhelp book")
    assert is_recommendation_request("5 mystery book for kids")


@pytest.mark.parametrize(
    "question",
    [
        "5 spy books",
        "5 mystery book for kids",
        "5 financial books",
        "5 selfhelp book",
        "Give me five books about investing",
        "Recommend good gifts for a friend",
        "Show 5 websites for calculus",
        "Top 10 movies for kids",
    ],
)
def test_all_curated_lists_use_the_recommendation_path(question: str) -> None:
    assert route_request(question, []).kind == RequestKind.RECOMMENDATION


@pytest.mark.parametrize(
    "question",
    [
        "Who created C++?",
        "Who is A. P. J. Abdul Kalam?",
        "List five books written by A. P. J. Abdul Kalam",
        "Who was the first captain of the Indian cricket team?",
    ],
)
def test_factual_questions_keep_the_evidence_verification_path(question: str) -> None:
    assert route_request(question, []).kind == RequestKind.FACTUAL


@pytest.mark.parametrize(
    "question",
    [
        "C++ or Java what is better?",
        "Which is better for backend development, Java or Python?",
        "Should I choose React or Angular?",
    ],
)
def test_advice_comparisons_do_not_receive_a_factual_reliability_score(question: str) -> None:
    assert route_request(question, []).kind == RequestKind.RECOMMENDATION


@pytest.mark.parametrize(
    "question",
    [
        "What is 9!",
        "Formula for integration of cosinex and derivative of tanx",
        "formula to calculate determinant of a 3x3 matrix",
        "Is 3+3=9?",
    ],
)
def test_math_requests_use_the_deterministic_path(question: str) -> None:
    assert route_request(question, []).kind == RequestKind.MATH


def test_basic_math_uses_deterministic_verification() -> None:
    answer = "The integral of cos(x) is sin(x) + C. The derivative of sin(x) is cos(x)."
    claims = verify_math_answer("What is integration of cosx and derivative of sinx", answer)
    assert len(claims) == 2
    assert all(claim.status == "supported" for claim in claims)


def test_factorials_are_recognized_as_math() -> None:
    assert is_math_question("What is 9!")
    claims = verify_math_answer("What is 9!", "9! equals 362,880.")
    assert claims[0].status == "supported"


def test_factorial_product_and_integration_formulas_are_checked() -> None:
    answer = (
        "4! equals 24. 5! equals 120. Their product equals 2,880. "
        "The integral of a constant k is kx + C. The integral of 1/x is ln|x| + C. "
        "The integral of e^x is e^x + C. The integral of sin(x) is -cos(x) + C. "
        "The integral of cos(x) is sin(x) + C."
    )
    claims = verify_math_answer("What is 4!*5!? And give basic integration formulas", answer)
    assert len(claims) == 8
    assert all(claim.status == "supported" for claim in claims)


def test_scaled_trigonometric_calculus_is_checked() -> None:
    answer = "The integral of 2sin(x) is -2cos(x) + C. The derivative of 4cos(x) is -4sin(x)."
    claims = verify_math_answer("What is integration of 2sinx and derivative of 4cosx", answer)
    assert len(claims) == 2
    assert all(claim.status == "supported" for claim in claims)


def test_three_by_three_determinant_formula_is_checked() -> None:
    answer = "det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)."
    claims = verify_math_answer("formula to calculate determinant of a 3x3 matrix", answer)
    assert claims[0].status == "supported"


def test_math_shorthand_is_normalized() -> None:
    assert normalize_math_shorthand("cosine x integration formula") == "cos(x) integration formula"
    assert normalize_math_shorthand("Derivation of secx") == "Derivation of sec(x)"


def test_math_follow_up_uses_previous_math_context() -> None:
    history = [ChatMessage(role="user", content="Formula for integration of cosinex and derivative of tanx")]
    assert resolve_math_follow_up("I asked for cosinex", history) == "What is the integration formula for cos(x)?"


def test_autobiography_follow_up_uses_last_book_title() -> None:
    history = [ChatMessage(role="user", content="Author of Wings of Fire")]
    assert resolve_contextual_question("I was asking about the autobiography one", history) == (
        "Who is the author of the autobiography titled Wings of Fire?"
    )


def test_autobiography_follow_up_uses_title_from_book_wording() -> None:
    history = [ChatMessage(role="user", content="Wings of Fire book")]
    assert resolve_contextual_question("The autobiography one", history) == (
        "Who is the author of the autobiography titled Wings of Fire?"
    )


def test_book_follow_up_is_routed_as_a_complete_factual_question() -> None:
    history = [ChatMessage(role="user", content="Wings of Fire book")]
    routed = route_request("The autobiography one", history)
    assert routed.kind == RequestKind.FACTUAL
    assert routed.question == "Who is the author of the autobiography titled Wings of Fire?"


def test_general_follow_up_resolves_pronouns_using_the_recent_topic() -> None:
    history = [
        ChatMessage(role="user", content="Who is Kalpana Chawla?"),
        ChatMessage(role="assistant", content="Kalpana Chawla was an astronaut."),
    ]
    assert resolve_contextual_question("When was she born?", history) == "When was Kalpana Chawla born?"


def test_general_follow_up_resolves_an_implicit_request_for_more_information() -> None:
    history = [ChatMessage(role="user", content="Who created the Python programming language?")]
    assert resolve_contextual_question("Tell me more", history) == "Provide more information about the Python programming language."


def test_general_follow_up_accepts_give_more_information_wording() -> None:
    history = [ChatMessage(role="user", content="Who created Python programming?")]
    assert resolve_contextual_question("Give more information", history) == "Provide more information about Python programming."


def test_vague_follow_up_without_history_requires_a_topic() -> None:
    routed = route_request("Tell me more", [])

    assert routed.kind == RequestKind.CONTEXT_REQUIRED


def test_achievement_question_supplies_the_topic_for_more_information() -> None:
    history = [ChatMessage(role="user", content="What are the achievements of Marie Curie?")]

    assert resolve_contextual_question("Tell me more", history) == "Provide more information about Marie Curie."


def test_bare_named_topic_supplies_context_for_elaboration() -> None:
    history = [ChatMessage(role="user", content="Marie Curie")]

    assert resolve_contextual_question("Elaborate", history) == "Provide more information about Marie Curie."


@pytest.mark.parametrize(
    "follow_up",
    [
        "Explain further",
        "Expand on this",
        "Could you provide more details?",
        "Share additional context",
        "Go deeper",
        "Continue please",
        "More about it",
    ],
)
def test_follow_up_synonyms_keep_the_previous_topic(follow_up: str) -> None:
    history = [ChatMessage(role="user", content="What are the achievements of Marie Curie?")]

    assert resolve_contextual_question(follow_up, history) == "Provide more information about Marie Curie."


def test_list_names_follow_up_keeps_the_original_list_subject() -> None:
    history = [
        ChatMessage(role="user", content="List the papers for GATE 2027"),
        ChatMessage(role="assistant", content="GATE 2027 has 30 test papers."),
    ]

    resolved = resolve_contextual_question("List only the names", history)

    assert resolved == "List only the names of papers for GATE 2027."
    assert route_request("List only the names", history).kind == RequestKind.FACTUAL


def test_contextual_alternative_request_keeps_the_previous_topic() -> None:
    history = [
        ChatMessage(role="user", content="Who created Vercel?"),
        ChatMessage(role="assistant", content="Guillermo Rauch founded Vercel."),
    ]

    routed = route_request("Give alternative for this", history)

    assert routed.question == "Recommend alternatives to Vercel."
    assert routed.kind == RequestKind.RECOMMENDATION


def test_another_author_becomes_a_recommendation_based_on_the_previous_answer() -> None:
    history = [
        ChatMessage(role="user", content="Author of Life After Life"),
        ChatMessage(role="assistant", content="The author of Life After Life is Kate Atkinson."),
    ]
    routed = route_request("Another author", history)
    assert routed.kind == RequestKind.RECOMMENDATION
    assert routed.question == "Recommend another author similar to Kate Atkinson."


def test_secant_derivative_is_checked() -> None:
    answer = "The derivative of sec(x) is sec(x)tan(x)."
    claims = verify_math_answer("Derivation of secx", answer)
    assert claims[0].status == "supported"


def test_cotangent_integral_is_checked() -> None:
    claims = verify_math_answer("integration of cotx", "ln(|sin(x)|) + C")
    assert claims[0].status == "supported"


def test_arithmetic_is_checked_without_web_evidence() -> None:
    claims = verify_math_answer("Is 2+2=7?", "No, two plus two equals four.")
    assert claims[0].status == "supported"
