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


def _number_label(coefficient: str) -> str:
    return "" if coefficient in {"", "1"} else coefficient


def _linear_trig_checks(question: str, answer: str) -> list[ClaimAssessment]:
    normalized_question = _normalized(question)
    normalized_answer = _normalized(answer)
    checks: list[ClaimAssessment] = []
    patterns = (
        (
            r"(?:integral|integration|integrate)(?:of)?(?P<coefficient>\d*)sin\(?x\)?",
            "The integral of {coefficient}sin(x) is -{coefficient}cos(x) + C.",
            lambda coefficient: f"-{_number_label(coefficient)}cos(x)+c",
        ),
        (
            r"(?:integral|integration|integrate)(?:of)?(?P<coefficient>\d*)cos\(?x\)?",
            "The integral of {coefficient}cos(x) is {coefficient}sin(x) + C.",
            lambda coefficient: f"{_number_label(coefficient)}sin(x)+c",
        ),
        (
            r"(?:integral|integration|integrate)(?:of)?(?P<coefficient>\d*)cot\(?x\)?",
            "The integral of {coefficient}cot(x) is {coefficient}ln|sin(x)| + C.",
            lambda coefficient: f"{_number_label(coefficient)}ln(|sin(x)|)+c",
        ),
        (
            r"(?:derivative|differentiate|derivation)(?:of)?(?P<coefficient>\d*)sin\(?x\)?",
            "The derivative of {coefficient}sin(x) is {coefficient}cos(x).",
            lambda coefficient: f"{_number_label(coefficient)}cos(x)",
        ),
        (
            r"(?:derivative|differentiate|derivation)(?:of)?(?P<coefficient>\d*)cos\(?x\)?",
            "The derivative of {coefficient}cos(x) is -{coefficient}sin(x).",
            lambda coefficient: f"-{_number_label(coefficient)}sin(x)",
        ),
        (
            r"(?:derivative|differentiate|derivation)(?:of)?(?P<coefficient>\d*)tan\(?x\)?",
            "The derivative of {coefficient}tan(x) is {coefficient}sec²(x).",
            lambda coefficient: f"{_number_label(coefficient)}sec²(x)",
        ),
        (
            r"(?:derivative|differentiate|derivation)(?:of)?(?P<coefficient>\d*)sec\(?x\)?",
            "The derivative of {coefficient}sec(x) is {coefficient}sec(x)tan(x).",
            lambda coefficient: f"{_number_label(coefficient)}sec(x)tan(x)",
        ),
        (
            r"(?:derivative|differentiate|derivation)(?:of)?(?P<coefficient>\d*)csc\(?x\)?",
            "The derivative of {coefficient}csc(x) is -{coefficient}csc(x)cot(x).",
            lambda coefficient: f"-{_number_label(coefficient)}csc(x)cot(x)",
        ),
        (
            r"(?:derivative|differentiate|derivation)(?:of)?(?P<coefficient>\d*)cot\(?x\)?",
            "The derivative of {coefficient}cot(x) is -{coefficient}csc²(x).",
            lambda coefficient: f"-{_number_label(coefficient)}csc²(x)",
        ),
    )
    for pattern, label, expected in patterns:
        match = re.search(pattern, normalized_question)
        if not match:
            continue
        coefficient = match.group("coefficient") or "1"
        rendered = _number_label(coefficient)
        checks.append(
            _assessment(
                label.format(coefficient=rendered),
                expected(coefficient) in normalized_answer,
            )
        )
    return checks


def _determinant_check(question: str, answer: str) -> list[ClaimAssessment]:
    dimensions = r"(?:3\s*[x×]\s*3|3by3)"
    if not re.search(
        rf"\b{dimensions}\b.*\bdeterminant|\bdeterminant\b.*\b{dimensions}\b",
        question,
        flags=re.IGNORECASE,
    ):
        return []
    normalized = _normalized(answer)
    valid = all(part in normalized for part in ("a(ei-fh)", "-b(di-fg)", "+c(dh-eg)"))
    return [_assessment("det(A) = a(ei − fh) − b(di − fg) + c(dh − eg).", valid)]


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

    assessments.extend(_linear_trig_checks(question, answer))
    assessments.extend(_determinant_check(question, answer))

    assessments.extend(_factorial_checks(question, answer))
    assessments.extend(_basic_integration_checks(question, answer))

    arithmetic = _arithmetic_expression(question)
    if arithmetic:
        value, expression = arithmetic
        rendered = str(int(value)) if value.is_integer() else str(value)
        assessments.append(_assessment(f"{expression} = {rendered}.", _answer_has_number(answer, value)))

    return assessments
