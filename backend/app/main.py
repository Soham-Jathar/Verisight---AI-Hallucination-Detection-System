from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import analyze
from app.schemas import HealthResponse, RootResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend for an evidence-grounded AI hallucination and uncertainty detector.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status for local and deployment health checks."""
    return HealthResponse()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(message="Hallucination Detection API is running")
