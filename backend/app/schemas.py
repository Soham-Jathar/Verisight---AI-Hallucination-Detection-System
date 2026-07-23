from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class VerificationMode(str, Enum):
    WEB = "web"
    DOCUMENT = "document"
    HYBRID = "hybrid"


class LLMProvider(str, Enum):
    EVIDENCE = "evidence"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    COMPARE = "compare"


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    mode: VerificationMode = VerificationMode.WEB
    provider: LLMProvider = LLMProvider.EVIDENCE


class EvidenceSource(BaseModel):
    title: str
    url: str
    snippet: str


class ClaimAssessment(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ProviderInfo(BaseModel):
    id: LLMProvider
    label: str
    model: str | None = None
    configured: bool


class ModelAnalysis(BaseModel):
    provider: LLMProvider
    model: str
    answer: str
    claims: list[ClaimAssessment] = Field(default_factory=list)
    reliability_score: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    question: str
    mode: VerificationMode
    provider: LLMProvider
    stage: str
    message: str
    answer: str | None = None
    model: str | None = None
    evidence: list[EvidenceSource] = Field(default_factory=list)
    claims: list[ClaimAssessment] = Field(default_factory=list)
    reliability_score: float | None = Field(default=None, ge=0.0, le=1.0)
    comparisons: list[ModelAnalysis] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class RootResponse(BaseModel):
    message: str
