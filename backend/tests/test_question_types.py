from app.services.math_verifier import verify_math_answer
from app.services.question_types import is_math_question, is_recommendation_request


def test_gift_ideas_are_not_fact_checked() -> None:
    assert is_recommendation_request("List 5 gift ideas for professor")


def test_basic_math_uses_deterministic_verification() -> None:
    answer = "The integral of cos(x) is sin(x) + C. The derivative of sin(x) is cos(x)."
    claims = verify_math_answer("What is integration of cosx and derivative of sinx", answer)
    assert len(claims) == 2
    assert all(claim.status == "supported" for claim in claims)


def test_arithmetic_is_checked_without_web_evidence() -> None:
    claims = verify_math_answer("Is 2+2=7?", "No, two plus two equals four.")
    assert claims[0].status == "supported"
