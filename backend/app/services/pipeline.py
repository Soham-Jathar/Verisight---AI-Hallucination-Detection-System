from __future__ import annotations

from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, LLMProvider, ModelAnalysis, VerificationMode
from app.services.generator import generate_answer, provider_info
from app.services.retrieval import retrieve_web_evidence
from app.services.verifier import reliability_score, verify_claims


async def run_analysis(request: AnalyzeRequest, *, settings: Settings) -> AnalyzeResponse:
    if request.mode != VerificationMode.WEB:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Verification mode '{request.mode.value}' is not available yet.",
        )

    evidence = await retrieve_web_evidence(request.question, settings=settings)
    selected = [request.provider]
    if request.provider == LLMProvider.COMPARE:
        selected = [
            provider.id
            for provider in provider_info(settings)
            if provider.id != LLMProvider.EVIDENCE and provider.configured
        ]
        if len(selected) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comparison requires at least two configured LLM providers.",
            )

    analyses: list[ModelAnalysis] = []
    for provider in selected:
        try:
            answer, model = await generate_answer(
                request.question,
                evidence,
                settings=settings,
                provider=provider,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        claims = verify_claims(answer, evidence)
        analyses.append(
            ModelAnalysis(
                provider=provider,
                model=model,
                answer=answer,
                claims=claims,
                reliability_score=reliability_score(claims),
            )
        )

    primary = analyses[0]
    answer = primary.answer
    claims = primary.claims
    score = primary.reliability_score

    supported = sum(1 for claim in claims if claim.status == "supported")
    uncertain = sum(1 for claim in claims if claim.status == "uncertain")
    unsupported = sum(1 for claim in claims if claim.status == "unsupported")

    if not evidence:
        message = (
            "No web evidence was retrieved. The answer could not be verified against sources."
        )
    else:
        message = (
            f"Analyzed {len(claims)} claim(s) using {len(evidence)} evidence source(s). "
            f"Supported: {supported}, uncertain: {uncertain}, unsupported: {unsupported}. "
            f"Reliability score: {score:.2f}. "
            f"Generation source: {primary.provider.value}."
        )

    return AnalyzeResponse(
        question=request.question,
        mode=request.mode,
        provider=request.provider,
        stage="complete",
        message=message,
        answer=answer,
        model=primary.model,
        evidence=evidence,
        claims=claims,
        reliability_score=score,
        comparisons=analyses if request.provider == LLMProvider.COMPARE else [],
    )
