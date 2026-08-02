from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


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


class RequestKind(str, Enum):
    """The verification strategy selected before a request reaches retrieval."""

    FACTUAL = "factual"
    MATH = "math"
    RECOMMENDATION = "recommendation"


@dataclass(frozen=True)
class RoutedRequest:
    """A user request after shorthand and conversational context are resolved."""

    question: str
    kind: RequestKind


_COUNT_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|a\s+few|some|\d+)"
_CURATED_ITEMS = (
    r"(?:books?|novels?|movies?|films?|shows?|series|songs?|podcasts?|"
    r"gifts?|presents?|ideas?|sites?|websites?|resources|links|apps?|"
    r"courses?|restaurants?|places?|destinations?)"
)
_COUNTED_LIST = re.compile(
    rf"^\s*(?:(?:give|list|show|suggest|recommend)\s+(?:me\s+)?)?"
    rf"{_COUNT_WORDS}\s+(?:[\w-]+\s+){{0,5}}{_CURATED_ITEMS}\b",
    re.IGNORECASE,
)
_CURATION_LANGUAGE = re.compile(
    r"\b(?:recommend|suggest|gift ideas?|what should i (?:buy|get|give)|"
    r"must[- ]?read|books? to read|reading list|best|top)\b",
    re.IGNORECASE,
)
_RESOURCE_REQUEST = re.compile(
    r"\b(?:give|list|show|suggest|recommend)\s+(?:me\s+)?(?:\d+|one|two|three|four|five|some|a\s+few)\s+"
    r"(?:sites?|websites?|resources|links)\b",
    re.IGNORECASE,
)
_FACT_LIST_QUALIFIER = re.compile(
    r"\b(?:written|created|made|published|developed|directed|authored)\s+by\b",
    re.IGNORECASE,
)


def _looks_like_recommendation(question: str) -> bool:
    """Identify curation requests without treating factual list questions as advice."""
    normalized = question.strip()
    if not normalized:
        return False

    # "List five books written by ..." is a factual catalogue request, not a
    # preference-based recommendation. It should still be evidence-checked.
    if _FACT_LIST_QUALIFIER.search(normalized):
        return False

    return bool(
        _COUNTED_LIST.search(normalized)
        or _CURATION_LANGUAGE.search(normalized)
        or _RESOURCE_REQUEST.search(normalized)
        or re.search(
            r"\b(?:sites?|websites?|resources|links)\s+(?:for|related to|about)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def is_recommendation_request(question: str) -> bool:
    """Return true for preference-based requests that should not be fact-checked."""
    return _looks_like_recommendation(question)


def is_math_question(question: str) -> bool:
    """Recognise questions better checked with deterministic math rules."""
    normalized = question.lower()
    return bool(
        re.search(
            r"\b(?:integral|integration|integrate|derivative|differentiate|derivation|"
            r"solve|calculate|evaluate|determinant)\b",
            normalized,
        )
        or re.search(r"\b\d+(?:\.\d+)?\s*[+*/-]\s*\d+(?:\.\d+)?\b", normalized)
        or bool(re.search(r"\b\d+\s*!", normalized))
        or bool(
            re.search(
                r"\b(?:sin|cos|tan|sec|csc|cot)\s*(?:\(\s*x\s*\)|x)(?=$|\s|[.,!?])",
                normalized,
            )
        )
    )


def normalize_math_shorthand(question: str) -> str:
    normalized = question
    for shorthand, expression in MATH_SHORTHANDS.items():
        normalized = re.sub(
            rf"\b{re.escape(shorthand)}\b",
            expression,
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def resolve_math_follow_up(question: str, history) -> str:
    """Use the immediately preceding math request to interpret a brief correction."""
    current = normalize_math_shorthand(question)
    match = re.match(
        r"\s*(?:i\s+)?(?:asked\s+for|meant)\s+(cos\(x\)|sin\(x\)|tan\(x\))\s*[.!?]*$",
        current,
        flags=re.IGNORECASE,
    )
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


def _recent_book_title(history) -> str | None:
    for message in reversed(history or []):
        if getattr(message, "role", None) != "user":
            continue
        content = message.content.strip()
        match = re.search(r"\bauthor\s+of\s+(.+?)[?!.,]*$", content, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        book_match = re.match(
            r"\s*(.+?)\s+(?:book|novel|series|autobiography)\s*[?!.,]*$",
            content,
            flags=re.IGNORECASE,
        )
        if book_match:
            return book_match.group(1).strip()
    return None


def _topic_from_question(question: str) -> str | None:
    """Extract the subject of the most recent user question for a follow-up."""
    cleaned = question.strip().rstrip("?!.")
    patterns = (
        r"^\s*who\s+(?:is|was)\s+(.+?)(?=\s+(?:and|with|including)\b|$)",
        r"^\s*who\s+(?:created|invented|designed|founded)\s+(.+?)(?=\s+(?:and|with|including)\b|$)",
        r"^\s*(?:tell me|give me information|explain)\s+(?:about\s+)?(.+?)(?=\s+(?:and|with|including)\b|$)",
        r"^\s*what\s+is\s+(.+?)(?=\s+(?:and|with|including)\b|$)",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip(" ,")
            if len(topic) >= 2:
                return topic
    return None


def _recent_topic(history) -> str | None:
    for message in reversed(history or []):
        if getattr(message, "role", None) != "user":
            continue
        topic = _topic_from_question(message.content)
        if topic:
            return topic
    return None


def _recent_answered_person(history) -> str | None:
    """Find a named person from the preceding answer for 'another author' style requests."""
    for message in reversed(history or []):
        if getattr(message, "role", None) != "assistant":
            continue
        match = re.search(
            r"\b(?:is|was)\s+([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,3})\b",
            message.content,
        )
        if match:
            return match.group(1)
    return None


_MORE_INFORMATION_FOLLOW_UP = (
    r"(?:"
    r"(?:tell me|explain|elaborate)(?:\s+(?:more|further|in detail))?"
    r"|give(?:\s+me)?\s+(?:more\s+)?(?:information|details)"
    r"|more details"
    r"|more information"
    r"|what about (?:him|her|it|that|this)"
    r")"
)
_ALTERNATIVE_FOLLOW_UP = re.compile(
    r"^\s*(?:give|list|show|suggest|recommend)\s+(?:me\s+)?(?:an?\s+|some\s+)?"
    r"alternatives?\s+(?:for|to|of)\s+(?:it|this|that|these|those)\s*[.!?]*$",
    re.IGNORECASE,
)


def _looks_like_general_follow_up(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    return bool(
        re.search(r"\b(?:he|she|they|it|his|her|their|its|this|that|these|those)\b", normalized)
        or re.fullmatch(rf"{_MORE_INFORMATION_FOLLOW_UP}[.!?]*", normalized)
        or re.fullmatch(r"another author[.!?]*", normalized)
        or normalized.startswith(("and ", "also "))
    )


def _resolve_general_follow_up(question: str, history) -> str:
    """Make references in a short follow-up explicit for generation and search."""
    if not _looks_like_general_follow_up(question):
        return question
    if re.fullmatch(r"another author[.!?]*", question.strip(), flags=re.IGNORECASE):
        author = _recent_answered_person(history)
        if author:
            return f"Recommend another author similar to {author}."
        return "Recommend another author."
    topic = _recent_topic(history)
    if not topic:
        return question

    normalized = question.strip()
    if _ALTERNATIVE_FOLLOW_UP.fullmatch(normalized):
        # Make the referent and the user's intent explicit. This avoids a model
        # interpreting "Alternative for ..." as the name of an unrelated entity.
        return f"Recommend alternatives to {topic}."
    if re.fullmatch(rf"{_MORE_INFORMATION_FOLLOW_UP}[.!?]*", normalized, re.IGNORECASE):
        return f"Provide more information about {topic}."

    possessive = f"{topic}'s"
    resolved = re.sub(r"\b(?:his|her|their|its)\b", possessive, normalized, flags=re.IGNORECASE)
    resolved = re.sub(r"\b(?:he|she|they|it|this|that|these|those)\b", topic, resolved, flags=re.IGNORECASE)
    if resolved.lower().startswith("and "):
        resolved = resolved[4:].strip()
    elif resolved.lower().startswith("also "):
        resolved = resolved[5:].strip()
    return resolved


def resolve_contextual_question(question: str, history) -> str:
    """Resolve concise math, title, and conversational follow-ups."""
    resolved = resolve_math_follow_up(question, history)
    if resolved != normalize_math_shorthand(question):
        return resolved

    if not re.search(
        r"\b(?:autobiography|novel|series|book)\s+(?:one|book|version)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return _resolve_general_follow_up(resolved, history)

    title = _recent_book_title(history)
    if title:
        return f"Who is the author of the autobiography titled {title}?"
    return _resolve_general_follow_up(resolved, history)


def route_request(question: str, history) -> RoutedRequest:
    """Select one consistent generation/verification path for the complete request."""
    resolved = resolve_contextual_question(question, history)
    if is_math_question(resolved):
        return RoutedRequest(question=resolved, kind=RequestKind.MATH)
    if _looks_like_recommendation(resolved):
        return RoutedRequest(question=resolved, kind=RequestKind.RECOMMENDATION)
    return RoutedRequest(question=resolved, kind=RequestKind.FACTUAL)
