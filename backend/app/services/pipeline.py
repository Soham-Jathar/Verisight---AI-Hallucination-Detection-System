from __future__ import annotations

import asyncio

from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, CorrectedAnswer, EvidenceSource, LLMProvider, ModelAnalysis, VerificationMode
from app.services.generator import generate_answer, generate_correction, provider_info
from app.services.documents import document_evidence
from app.services.retrieval import retrieve_web_evidence
from app.services.verifier import (
    reliability_score,
    select_citations,
    select_verification_sources,
    verify_claims,
)


def _merge_evidence(*groups):
    merged = []
    indexes = {}
    for group in groups:
        for source in group:
            key = source.url or source.title.lower()
            if key in indexes:
                index = indexes[key]
                existing = merged[index]
                if source.snippet not in existing.snippet:
                    merged[index] = EvidenceSource(
                        title=existing.title,
                        url=existing.url,
                        snippet=f"{existing.snippet} {source.snippet}"[:3_000],
                    )
                continue
            indexes[key] = len(merged)
            merged.append(source)
    return merged


async def _expand_uncertain_claim_evidence(
    question: str,
    claims,
    evidence,
    *,
    settings: Settings,
):
    """Retrieve focused evidence only for claims the first pass could not settle."""
    uncertain = [claim.claim for claim in claims if claim.status == "uncertain"][:4]
    if not uncertain:
        return evidence

    searches = [
        retrieve_web_evidence(
            f"{question} Factual claim to verify: {claim}",
            settings=settings,
        )
        for claim in uncertain
    ]
    results = await asyncio.gather(*searches, return_exceptions=True)
    successful = [result for result in results if isinstance(result, list)]
    return _merge_evidence(evidence, *successful)


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
    primary_evidence = list(evidence)
    shared_evidence = list(evidence)
    for index, provider in enumerate(selected):
        try:
            answer, model = await generate_answer(
                request.question,
                shared_evidence,
                settings=settings,
                provider=provider,
                history=request.history,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        analysis_evidence = list(shared_evidence)
        claims = verify_claims(answer, analysis_evidence) if request.verify else []
        if (
            request.verify
            and request.mode in {VerificationMode.WEB, VerificationMode.HYBRID}
            and any(claim.status == "uncertain" for claim in claims)
        ):
            analysis_evidence = await _expand_uncertain_claim_evidence(
                request.question,
                claims,
                analysis_evidence,
                settings=settings,
            )
            claims = verify_claims(answer, analysis_evidence)
        if index == 0:
            primary_evidence = analysis_evidence
            shared_evidence = analysis_evidence
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
    visible_evidence = select_verification_sources(claims, primary_evidence) if request.verify else []
    if request.verify and primary_evidence and unsupported:
        try:
            corrected_answer, _ = await generate_correction(
                request.question,
                primary_evidence,
                settings=settings,
                provider=primary.provider,
            )
            correction = CorrectedAnswer(
                answer=corrected_answer,
                citations=select_citations(corrected_answer, primary_evidence),
            )
        except ValueError:
            # A correction is helpful but must never hide the original analysis result.
            correction = None

    if not request.verify:
        message = "Answer generated without verification."
    elif not primary_evidence:
        message = (
            "No evidence was retrieved. The answer could not be verified against sources."
        )
    else:
        message = (
            f"Analyzed {len(claims)} claim(s) using {len(visible_evidence)} evidence source(s). "
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
        evidence=visible_evidence,
        claims=claims,
        reliability_score=score,
        correction=correction,
        comparisons=analyses if request.provider == LLMProvider.COMPARE else [],
    )
