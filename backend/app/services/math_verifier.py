from __future__ import annotations

import math
import re

from app.schemas import ClaimAssessment


NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _normalized(text: str) -> str:
    return (
        text.lower()
        .replace("\\", "")
        .replace(" ", "")
        .replace("{", "")
        .replace("}", "")
        .replace("$", "")
    )


def _assessment(claim: str, correct: bool) -> ClaimAssessment:
    return ClaimAssessment(
        claim=claim,
        status="supported" if correct else "unsupported",
        confidence=0.99,
        rationale=(
            "A deterministic mathematical rule confirms this statement."
            if correct
            else "A deterministic mathematical rule contradicts this statement."
        ),
    )


def _contains_integral_cos(answer: str) -> bool:
    normalized = _normalized(answer)
    return bool(re.search(r"sin\(?x\)?\+?(?:c|constant)", normalized))


def _contains_derivative_sin(answer: str) -> bool:
    normalized = _normalized(answer)
    return bool(re.search(r"(?:derivativeof)?sin\(?x\)?.{0,32}(?:is|=)?cos\(?x\)?", normalized))


def _arithmetic_expression(question: str) -> tuple[float, str] | None:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*([+*/-])\s*(\d+(?:\.\d+)?)\b", question)
    if not match:
        return None
    left, operator, right = match.groups()
    first, second = float(left), float(right)
    if operator == "+":
        result = first + second
    elif operator == "-":
        result = first - second
    elif operator == "*":
        result = first * second
    else:
        if second == 0:
            return None
        result = first / second
    expression = f"{left} {operator} {right}"
    return result, expression


def _answer_has_number(answer: str, value: float) -> bool:
    compact = _normalized(answer)
    if value.is_integer():
        integer = int(value)
        return bool(re.search(rf"\b{integer}\b", answer)) or NUMBER_WORDS.get(integer, "") in compact
    return str(value) in compact


def _contains_all(answer: str, *parts: str) -> bool:
    normalized = _normalized(answer)
    return all(part in normalized for part in parts)


def _factorial_checks(question: str, answer: str) -> list[ClaimAssessment]:
    values = [int(value) for value in re.findall(r"\b(\d+)\s*!", question)]
    assessments: list[ClaimAssessment] = []
    for value in dict.fromkeys(values):
        if value > 20:
            continue
        result = math.factorial(value)
        assessments.append(_assessment(f"{value}! = {result:,}.", _answer_has_number(answer, float(result))))

    product = re.search(r"\b(\d+)\s*!\s*[×*x]\s*(\d+)\s*!", question, flags=re.IGNORECASE)
    if product:
        first, second = (int(value) for value in product.groups())
        if first <= 20 and second <= 20:
            result = math.factorial(first) * math.factorial(second)
            assessments.append(
                _assessment(f"{first}! × {second}! = {result:,}.", _answer_has_number(answer, float(result)))
            )
    return assessments


def _basic_integration_checks(question: str, answer: str) -> list[ClaimAssessment]:
    if not re.search(r"\bbasic\s+integration\s+formulas?\b", question, flags=re.IGNORECASE):
        return []
    return [
        _assessment("The integral of a constant k is kx + C.", _contains_all(answer, "kx+c")),
        _assessment("The integral of 1/x is ln|x| + C.", _contains_all(answer, "ln|x|+c")),
        _assessment("The integral of e^x is e^x + C.", _contains_all(answer, "e^x+c")),
        _assessment("The integral of sin(x) is -cos(x) + C.", _contains_all(answer, "-cos(x)+c")),
        _assessment("The integral of cos(x) is sin(x) + C.", _contains_all(answer, "sin(x)+c")),
    ]


def verify_math_answer(question: str, answer: str) -> list[ClaimAssessment]:
    """Verify a small, transparent set of arithmetic and calculus facts locally."""
    normalized_question = _normalized(question)
    assessments: list[ClaimAssessment] = []

    if re.search(r"(?:integral|integration|integrate).{0,30}cos\(?x\)?", normalized_question):
        assessments.append(_assessment("The integral of cos(x) is sin(x) + C.", _contains_integral_cos(answer)))
    if re.search(r"(?:derivative|differentiate).{0,30}sin\(?x\)?", normalized_question):
        assessments.append(_assessment("The derivative of sin(x) is cos(x).", _contains_derivative_sin(answer)))

    assessments.extend(_factorial_checks(question, answer))
    assessments.extend(_basic_integration_checks(question, answer))

    arithmetic = _arithmetic_expression(question)
    if arithmetic:
        value, expression = arithmetic
        rendered = str(int(value)) if value.is_integer() else str(value)
        assessments.append(_assessment(f"{expression} = {rendered}.", _answer_has_number(answer, value)))

    return assessments
