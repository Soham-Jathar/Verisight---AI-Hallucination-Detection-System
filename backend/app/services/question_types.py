from __future__ import annotations

import re


MATH_SHORTHANDS = {
    "cosinex": "cos(x)",
    "cosine x": "cos(x)",
    "sinex": "sin(x)",
    "sine x": "sin(x)",
    "tanx": "tan(x)",
    "tangent x": "tan(x)",
    "secx": "sec(x)",
    "secant x": "sec(x)",
    "cosecx": "csc(x)",
    "cosecant x": "csc(x)",
    "cotx": "cot(x)",
    "cotangent x": "cot(x)",
}


def is_recommendation_request(question: str) -> bool:
    """Return true for preference-based requests that cannot be fact-checked."""
    normalized = question.lower()
    patterns = (
        r"\bgift ideas?\b",
        r"\brecommend(?:ation)?s?\b",
        r"\bwhat should i (?:buy|get|give)\b",
        r"\b(?:best|good) (?:gift|present)\b",
        r"\b(?:give|list|show)\s+\d+\s+(?:sites|websites|resources|links)\b",
        r"\b(?:sites|websites|resources|links)\s+(?:for|related to|about)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_math_question(question: str) -> bool:
    """Recognise questions better checked with deterministic math rules."""
    normalized = question.lower()
    return bool(
        re.search(r"\b(?:integral|integration|integrate|derivative|differentiate|derivation|solve|calculate|evaluate|determinant)\b", normalized)
        or re.search(r"\b\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?\b", normalized)
        or bool(re.search(r"\b\d+\s*!", normalized))
        or bool(re.search(r"\b(?:sin|cos|tan|sec|csc|cot)\s*(?:\(\s*x\s*\)|x)(?=$|\s|[.,!?])", normalized))
    )


def normalize_math_shorthand(question: str) -> str:
    normalized = question
    for shorthand, expression in MATH_SHORTHANDS.items():
        normalized = re.sub(rf"\b{re.escape(shorthand)}\b", expression, normalized, flags=re.IGNORECASE)
    return normalized


def resolve_math_follow_up(question: str, history) -> str:
    """Use the immediately preceding math request to interpret a brief correction."""
    current = normalize_math_shorthand(question)
    match = re.match(r"\s*(?:i\s+)?(?:asked\s+for|meant)\s+(cos\(x\)|sin\(x\)|tan\(x\))\s*[.!?]*$", current, flags=re.IGNORECASE)
    if not match:
        return current

    previous_questions = [
        normalize_math_shorthand(message.content)
        for message in reversed(history or [])
        if getattr(message, "role", None) == "user"
    ]
    previous = next((item for item in previous_questions if is_math_question(item)), "")
    expression = match.group(1)
    if re.search(r"\b(?:integral|integration|integrate)\b", previous, flags=re.IGNORECASE):
        return f"What is the integration formula for {expression}?"
    if re.search(r"\b(?:derivative|differentiate)\b", previous, flags=re.IGNORECASE):
        return f"What is the derivative of {expression}?"
    return current
