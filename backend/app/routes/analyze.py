from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, ProviderInfo
from app.services.generator import provider_info
from app.services.pipeline import run_analysis

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/providers", response_model=list[ProviderInfo])
def providers(settings: Settings = Depends(get_settings)) -> list[ProviderInfo]:
    return provider_info(settings)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    return await run_analysis(request, settings=settings)
