from __future__ import annotations

import asyncio
import re

from app.config import Settings
from app.schemas import ChatMessage, EvidenceSource, LLMProvider
from app.services.generator import generate_answer


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def _similarity(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


async def estimate_uncertainty(
    question: str,
    primary_answer: str,
    evidence: list[EvidenceSource],
    *,
    settings: Settings,
    provider: LLMProvider,
    history: list[ChatMessage] | None,
) -> float | None:
    """Estimate answer instability from two independent, higher-variance samples."""
    if provider == LLMProvider.EVIDENCE:
        return None

    try:
        samples = await asyncio.gather(
            *[
                generate_answer(
                    question,
                    evidence,
                    settings=settings,
                    provider=provider,
                    history=history,
                    temperature=0.65,
                )
                for _ in range(2)
            ]
        )
    except ValueError:
        return None

    answers = [primary_answer, *(answer for answer, _ in samples)]
    similarities = [
        _similarity(answers[index], answers[other_index])
        for index in range(len(answers))
        for other_index in range(index + 1, len(answers))
    ]
    if not similarities:
        return None
    return round(1 - sum(similarities) / len(similarities), 2)
