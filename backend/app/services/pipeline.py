from __future__ import annotations

from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, CorrectedAnswer, LLMProvider, ModelAnalysis, VerificationMode
from app.services.generator import generate_answer, generate_correction, provider_info
from app.services.documents import document_evidence
from app.services.retrieval import retrieve_web_evidence
from app.services.verifier import reliability_score, select_citations, verify_claims


async def run_analysis(request: AnalyzeRequest, *, settings: Settings) -> AnalyzeResponse:
    evidence = []
    if request.verify and request.mode in {VerificationMode.DOCUMENT, VerificationMode.HYBRID}:
        if not request.document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload a PDF before using document or hybrid verification.",
            )
        evidence.extend(document_evidence(request.document_id, request.question))
    if request.verify and request.mode in {VerificationMode.WEB, VerificationMode.HYBRID}:
        try:
            evidence.extend(await retrieve_web_evidence(request.question, settings=settings))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
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
                history=request.history,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        claims = verify_claims(answer, evidence) if request.verify else []
        analyses.append(
            ModelAnalysis(
                provider=provider,
                model=model,
                answer=answer,
                claims=claims,
                reliability_score=reliability_score(claims) if request.verify else None,
            )
        )

    primary = analyses[0]
    answer = primary.answer
    claims = primary.claims
    score = primary.reliability_score

    supported = sum(1 for claim in claims if claim.status == "supported")
    uncertain = sum(1 for claim in claims if claim.status == "uncertain")
    unsupported = sum(1 for claim in claims if claim.status == "unsupported")
    correction = None
    if request.verify and evidence and unsupported:
        try:
            corrected_answer, _ = await generate_correction(
                request.question,
                evidence,
                settings=settings,
                provider=primary.provider,
            )
            correction = CorrectedAnswer(
                answer=corrected_answer,
                citations=select_citations(corrected_answer, evidence),
            )
        except ValueError:
            # A correction is helpful but must never hide the original analysis result.
            correction = None

    if not request.verify:
        message = "Answer generated without verification."
    elif not evidence:
        message = (
            "No evidence was retrieved. The answer could not be verified against sources."
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
        correction=correction,
        comparisons=analyses if request.provider == LLMProvider.COMPARE else [],
    )
