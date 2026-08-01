from __future__ import annotations

import re


def is_recommendation_request(question: str) -> bool:
    """Return true for preference-based requests that cannot be fact-checked."""
    normalized = question.lower()
    patterns = (
        r"\bgift ideas?\b",
        r"\brecommend(?:ation)?s?\b",
        r"\bwhat should i (?:buy|get|give)\b",
        r"\b(?:best|good) (?:gift|present)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_math_question(question: str) -> bool:
    """Recognise questions better checked with deterministic math rules."""
    normalized = question.lower()
    return bool(
        re.search(r"\b(?:integral|integration|integrate|derivative|differentiate|solve|calculate|evaluate)\b", normalized)
        or re.search(r"\b\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?\b", normalized)
        or bool(re.search(r"\b\d+\s*!", normalized))
    )
