from __future__ import annotations

import httpx

from app.config import Settings
from app.schemas import ChatMessage, EvidenceSource, LLMProvider, ProviderInfo
from app.services.retrieval import build_evidence_answer


PROVIDER_LABELS = {
    LLMProvider.EVIDENCE: "Evidence-backed baseline",
    LLMProvider.GEMINI: "Google Gemini",
    LLMProvider.GROQ: "Groq",
    LLMProvider.OPENROUTER: "OpenRouter",
    LLMProvider.COMPARE: "Compare configured models",
}


def provider_info(settings: Settings) -> list[ProviderInfo]:
    providers = [
        ProviderInfo(
            id=LLMProvider.EVIDENCE,
            label=PROVIDER_LABELS[LLMProvider.EVIDENCE],
            model="retrieval-summary",
            configured=True,
        ),
        ProviderInfo(
            id=LLMProvider.GEMINI,
            label=PROVIDER_LABELS[LLMProvider.GEMINI],
            model=settings.gemini_model,
            configured=bool(settings.gemini_api_key),
        ),
        ProviderInfo(
            id=LLMProvider.GROQ,
            label=PROVIDER_LABELS[LLMProvider.GROQ],
            model=settings.groq_model,
            configured=bool(settings.groq_api_key),
        ),
        ProviderInfo(
            id=LLMProvider.OPENROUTER,
            label=PROVIDER_LABELS[LLMProvider.OPENROUTER],
            model=settings.openrouter_model,
            configured=bool(settings.openrouter_api_key),
        ),
    ]
    configured_llms = sum(
        provider.configured
        for provider in providers
        if provider.id not in {LLMProvider.EVIDENCE, LLMProvider.COMPARE}
    )
    providers.append(
        ProviderInfo(
            id=LLMProvider.COMPARE,
            label=PROVIDER_LABELS[LLMProvider.COMPARE],
            model=None,
            configured=configured_llms >= 2,
        )
    )
    return providers


async def generate_answer(
    question: str,
    evidence: list[EvidenceSource],
    *,
    settings: Settings,
    provider: LLMProvider,
    history: list[ChatMessage] | None = None,
    temperature: float = 0.2,
) -> tuple[str, str]:
    if provider == LLMProvider.EVIDENCE:
        return build_evidence_answer(question, evidence), "retrieval-summary"
    if provider == LLMProvider.GEMINI:
        return await _generate_with_gemini(question, evidence, settings=settings, history=history, temperature=temperature)
    if provider == LLMProvider.GROQ:
        return await _generate_with_openai_compatible(
            question,
            evidence,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            model=settings.groq_model,
            settings=settings,
            history=history,
            temperature=temperature,
        )
    if provider == LLMProvider.OPENROUTER:
        return await _generate_with_openai_compatible(
            question,
            evidence,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=settings.openrouter_model,
            settings=settings,
            history=history,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported provider: {provider.value}")


async def generate_correction(
    question: str,
    evidence: list[EvidenceSource],
    *,
    settings: Settings,
    provider: LLMProvider,
) -> tuple[str, str]:
    """Create a replacement answer constrained to the retrieved evidence."""
    if provider == LLMProvider.EVIDENCE:
        return build_evidence_answer(question, evidence), "retrieval-summary"
    if provider == LLMProvider.GEMINI:
        return await _generate_with_gemini(question, evidence, settings=settings, history=None, correction=True)
    if provider == LLMProvider.GROQ:
        return await _generate_with_openai_compatible(
            question, evidence, api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1",
            model=settings.groq_model, settings=settings, history=None, correction=True,
        )
    if provider == LLMProvider.OPENROUTER:
        return await _generate_with_openai_compatible(
            question, evidence, api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1",
            model=settings.openrouter_model, settings=settings, history=None, correction=True,
        )
    raise ValueError(f"Unsupported provider: {provider.value}")


def _evidence_block(evidence: list[EvidenceSource]) -> str:
    return "\n".join(
        f"- {source.title}: {source.snippet}" for source in evidence[:5]
    ) or "No evidence was retrieved."


def _document_grounding_instruction(evidence: list[EvidenceSource]) -> str:
    """Tighten answer generation when the user selected document-only mode."""
    has_document = any(source.url.startswith("document://") for source in evidence)
    has_web_source = any(
        source.url.startswith(("http://", "https://"))
        for source in evidence
    )
    if has_document and not has_web_source:
        return (
            " The supplied evidence is an uploaded document and is authoritative for this answer. "
            "Answer only from that document. When it explicitly names a person, institute, organisation, "
            "date, number, or topic, state that exact value directly instead of replacing it with a vague description."
        )
    return ""


def _history_block(history: list[ChatMessage] | None) -> str:
    if not history:
        return "No previous conversation."
    recent = history[-8:]
    return "\n".join(
        f"{'User' if message.role == 'user' else 'Assistant'}: {message.content}"
        for message in recent
    )


def _provider_error(provider: str, error: httpx.HTTPStatusError) -> ValueError:
    """Turn an upstream API error into a safe message for the website."""
    detail = error.response.text.strip().replace("\n", " ")
    if len(detail) > 500:
        detail = f"{detail[:500]}..."
    return ValueError(
        f"{provider} request failed (HTTP {error.response.status_code}). "
        f"{detail or 'The provider did not return an error message.'}"
    )


def _connection_error(provider: str, error: httpx.RequestError) -> ValueError:
    """Make empty HTTP client errors understandable without exposing secrets."""
    detail = str(error).strip()
    cause = error.__cause__ or error.__context__
    cause_detail = str(cause).strip() if cause else ""
    error_type = type(error).__name__
    if detail:
        description = detail
    elif cause_detail:
        description = f"{type(cause).__name__}: {cause_detail}"
    else:
        description = error_type

    return ValueError(
        f"Could not reach {provider} ({description}). "
        "Check your internet connection and any VPN, proxy, or firewall, then retry."
    )


async def _generate_with_openai_compatible(
    question: str,
    evidence: list[EvidenceSource],
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    settings: Settings,
    history: list[ChatMessage] | None,
    correction: bool = False,
    temperature: float = 0.2,
) -> tuple[str, str]:
    if not api_key:
        raise ValueError("This provider has not been configured on the server.")

    evidence_block = _evidence_block(evidence)

    instruction = (
        "Write a concise corrected answer to the user's question. Use only the supplied evidence; "
        "do not add facts from general knowledge. If the evidence cannot support an answer, say so plainly. "
        "Do not mention the correction process, evidence, or citations."
        if correction
        else (
            "You are a concise, factual assistant. Answer the user's latest question "
            "directly using your general knowledge. The supplied evidence is useful "
            "context, but do not mention it or refuse solely because it is incomplete. "
            + _document_grounding_instruction(evidence)
            + " "
            "Do not start with phrases such as 'Based on the provided evidence'. "
            "Use short sentences with one independently verifiable fact per sentence; "
            "do not combine a role, date, achievement, and event into one sentence. "
            "Do not claim that a list is complete unless the supplied information explicitly establishes that. "
            "For factual answers that are not recommendations, provide no more than six independently "
            "verifiable sentences. If the user asks for more, provide the six most useful facts and say it is a selection. "
            "Do not limit a recommendation when the user requests a specific number of ideas. "
            "Do not use a company, product, organisation, or other named example in a factual answer unless "
            "the supplied evidence explicitly names it; prefer general supported categories instead. "
            "For a mathematical question, give only the requested result or formulas, with no extra identities. "
            "Use readable Unicode notation such as ∫, ×, ÷, √, π, and superscripts. Never use raw LaTex commands "
            "such as \\frac, \\int, or dollar-sign math delimiters."
        )
    )
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        *[
            {"role": message.role, "content": message.content}
            for message in (history or [])[-8:]
        ],
        {
            "role": "user",
            "content": f"Latest question: {question}\n\nEvidence context:\n{evidence_block}",
        },
    ]
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as error:
        raise _provider_error("LLM provider", error) from error
    except httpx.RequestError as error:
        raise _connection_error("the LLM provider", error) from error

    return data["choices"][0]["message"]["content"].strip(), model


async def _generate_with_gemini(
    question: str,
    evidence: list[EvidenceSource],
    *,
    settings: Settings,
    history: list[ChatMessage] | None,
    correction: bool = False,
    temperature: float = 0.2,
) -> tuple[str, str]:
    if not settings.gemini_api_key:
        raise ValueError("This provider has not been configured on the server.")

    instruction = (
        "Write a concise corrected answer to the user's question. Use only the supplied evidence; "
        "do not add facts from general knowledge. If the evidence cannot support an answer, say so plainly. "
        "Do not mention the correction process, evidence, or citations."
        if correction
        else (
            "You are a concise, factual assistant. Answer the user's question directly using "
            "your general knowledge. The supplied evidence is useful context, but do not mention "
            "it or refuse solely because it is incomplete. "
            + _document_grounding_instruction(evidence)
            + "Do not start with phrases such as 'Based on the provided evidence'. Use short sentences with one independently "
            "verifiable fact per sentence; do not combine a role, date, achievement, and event "
            "into one sentence. Do not claim that a list is complete unless the supplied information "
            "explicitly establishes that. For factual answers that are not recommendations, provide no more than six "
            "independently verifiable sentences. If the user asks for more, provide the six most useful facts and say "
            "it is a selection. Do not limit a recommendation when the user requests a specific number of ideas. "
            "Do not use a company, product, organisation, or other named example in a factual answer unless the "
            "supplied evidence explicitly names it; prefer general supported categories instead. "
            "For a mathematical question, give only the requested result or formulas, "
            "with no extra identities. Use readable Unicode notation such as ∫, ×, ÷, √, π, and superscripts. "
            "Never use raw LaTex commands such as \\frac, \\int, or dollar-sign math delimiters."
        )
    )
    prompt = (
        f"{instruction}\n\nConversation so far:\n{_history_block(history)}\n\n"
        f"Latest question: {question}\n\nEvidence context:\n{_evidence_block(evidence)}"
    )
    # Gemini can be slower than the OpenAI-compatible providers. Give it a
    # modestly longer response window, but do not silently retry a generation
    # because that could consume a second request from the user's quota.
    timeout = httpx.Timeout(settings.gemini_request_timeout_seconds)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as error:
        raise _provider_error("Gemini", error) from error
    except httpx.RequestError as error:
        raise _connection_error("Gemini", error) from error

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    answer = "".join(part.get("text", "") for part in parts).strip()
    if not answer:
        raise ValueError("Gemini returned no text response.")
    return answer, settings.gemini_model
