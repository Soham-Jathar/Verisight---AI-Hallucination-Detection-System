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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    mode: VerificationMode = VerificationMode.WEB
    provider: LLMProvider = LLMProvider.EVIDENCE
    verify: bool = True
    measure_uncertainty: bool = False
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    document_id: str | None = Field(default=None, max_length=80)


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: int = Field(ge=0)
    characters: int = Field(ge=0)


class EvidenceSource(BaseModel):
    title: str
    url: str
    snippet: str
    credibility: float = Field(default=0.55, ge=0.0, le=1.0)
    source_quality: str = "Web source"


class ClaimAssessment(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    citations: list[EvidenceSource] = Field(default_factory=list)


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
    reliability_score: float | None = Field(default=None, ge=0.0, le=1.0)


class CorrectedAnswer(BaseModel):
    answer: str
    citations: list[EvidenceSource] = Field(default_factory=list)


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
    uncertainty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    correction: CorrectedAnswer | None = None
    comparisons: list[ModelAnalysis] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class RootResponse(BaseModel):
    message: str
