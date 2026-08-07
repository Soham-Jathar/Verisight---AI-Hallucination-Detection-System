from __future__ import annotations

import asyncio

from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, CorrectedAnswer, EvidenceSource, LLMProvider, ModelAnalysis, VerificationMode
from app.services.generator import generate_answer, generate_correction, provider_info
from app.services.math_verifier import verify_math_answer
from app.services.math_notation import format_math_notation
from app.services.documents import document_evidence
from app.services.question_types import (
    RequestKind,
    route_request,
)
from app.services.retrieval import retrieve_web_evidence
from app.services.source_quality import enrich_source
from app.services.uncertainty import estimate_uncertainty
from app.services.verifier import (
    limit_factual_answer,
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
                        snippet=f"{existing.snippet} {source.snippet}"[:9_000],
                        credibility=max(existing.credibility, source.credibility),
                        source_quality=(
                            existing.source_quality
                            if existing.credibility >= source.credibility
                            else source.source_quality
                        ),
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
    document_id: str | None = None,
    include_web: bool = False,
):
    """Retrieve focused evidence only for claims the first pass could not settle."""
    uncertain = [claim.claim for claim in claims if claim.status == "uncertain"][:4]
    if not uncertain:
        return evidence

    document_results = [
        document_evidence(document_id, claim, limit=2)
        for claim in uncertain
    ] if document_id else []
    searches = [
        retrieve_web_evidence(
            f"{question} Factual claim to verify: {claim}",
            settings=settings,
        )
        for claim in uncertain
    ] if include_web else []
    results = await asyncio.gather(*searches, return_exceptions=True) if searches else []
    successful = [result for result in results if isinstance(result, list)]
    return _merge_evidence(evidence, *document_results, *successful)


async def run_analysis(request: AnalyzeRequest, *, settings: Settings) -> AnalyzeResponse:
    routed_request = route_request(request.question, request.history)
    analysis_question = routed_request.question
    recommendation_request = routed_request.kind == RequestKind.RECOMMENDATION
    math_question = routed_request.kind == RequestKind.MATH
    verification_applicable = request.verify and not recommendation_request
    evidence = []
    if verification_applicable and not math_question and request.mode in {VerificationMode.DOCUMENT, VerificationMode.HYBRID}:
        if not request.document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload a PDF before using document or hybrid verification.",
            )
        evidence.extend(document_evidence(request.document_id, analysis_question))
    if verification_applicable and not math_question and request.mode in {VerificationMode.WEB, VerificationMode.HYBRID}:
        try:
            evidence.extend(await retrieve_web_evidence(analysis_question, settings=settings))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
    evidence = [enrich_source(source) for source in evidence]
    selected = [request.provider]
    if request.provider == LLMProvider.COMPARE:
        selected = [
            provider.id
            for provider in provider_info(settings)
            if provider.id not in {LLMProvider.EVIDENCE, LLMProvider.COMPARE}
            and provider.configured
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
                analysis_question,
                shared_evidence,
                settings=settings,
                provider=provider,
                history=request.history,
            )
            if math_question:
                answer = format_math_notation(answer)
            elif verification_applicable and not recommendation_request:
                answer = limit_factual_answer(answer, question=analysis_question)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        analysis_evidence = list(shared_evidence)
        claims = (
            verify_math_answer(analysis_question, answer)
            if verification_applicable and math_question
            else verify_claims(answer, analysis_evidence, question=analysis_question)
            if verification_applicable
            else []
        )
        if (
            verification_applicable
            and not math_question
            and request.mode in {VerificationMode.WEB, VerificationMode.DOCUMENT, VerificationMode.HYBRID}
            and any(claim.status == "uncertain" for claim in claims)
        ):
            analysis_evidence = await _expand_uncertain_claim_evidence(
                analysis_question,
                claims,
                analysis_evidence,
                settings=settings,
                document_id=request.document_id if request.mode in {VerificationMode.DOCUMENT, VerificationMode.HYBRID} else None,
                include_web=request.mode in {VerificationMode.WEB, VerificationMode.HYBRID},
            )
            claims = verify_claims(answer, analysis_evidence, question=analysis_question)
        if index == 0:
            primary_evidence = analysis_evidence
            shared_evidence = analysis_evidence
        analyses.append(
            ModelAnalysis(
                provider=provider,
                model=model,
                answer=answer,
                claims=claims,
                reliability_score=(
                    reliability_score(claims) if verification_applicable and claims else None
                ),
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
    uncertainty_score = None
    visible_evidence = select_verification_sources(claims, primary_evidence) if verification_applicable else []
    if verification_applicable and not visible_evidence:
        # A vague answer may not have a claim-level citation, but document mode
        # should still show the uploaded file that was actually consulted.
        visible_evidence = [
            source
            for source in primary_evidence
            if source.url.startswith("document://")
        ][:1]
    if verification_applicable and not math_question and primary_evidence and unsupported:
        try:
            corrected_answer, _ = await generate_correction(
                request.question,
                primary_evidence,
                settings=settings,
                provider=primary.provider,
            )
            correction_citations = select_citations(corrected_answer, primary_evidence)
            # A correction without a directly relevant citation would look
            # authoritative while being no safer than the original answer.
            if correction_citations:
                correction = CorrectedAnswer(
                    answer=corrected_answer,
                    citations=correction_citations,
                )
        except ValueError:
            # A correction is helpful but must never hide the original analysis result.
            correction = None

    if request.measure_uncertainty:
        uncertainty_score = await estimate_uncertainty(
            analysis_question,
            answer,
            primary_evidence,
            settings=settings,
            provider=primary.provider,
            history=request.history,
        )

    if not request.verify:
        message = "Answer generated without verification."
    elif recommendation_request:
        message = "Verification is not applicable to preference-based recommendations."
    elif math_question and not claims:
        message = "This mathematical answer could not be checked by the available deterministic rules."
    elif not claims:
        message = (
            "No externally verifiable factual claims were found in this response, "
            "so no reliability score was calculated."
        )
    elif math_question:
        message = (
            f"Analyzed {len(claims)} mathematical statement(s) using deterministic rules. "
            f"Supported: {supported}, unsupported: {unsupported}. Reliability score: {score:.2f}."
        )
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
        if uncertainty_score is not None:
            message += f" Estimated uncertainty: {uncertainty_score:.2f}."

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
        uncertainty_score=uncertainty_score,
        correction=correction,
        comparisons=analyses if request.provider == LLMProvider.COMPARE else [],
    )
