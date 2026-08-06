"""Local-only endpoint for project evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def _latest_report_directory() -> Path:
    runs = sorted(
        (path for path in RESULTS_DIR.iterdir() if path.is_dir() and (path / "metrics.json").exists()),
        key=lambda path: path.name,
        reverse=True,
    ) if RESULTS_DIR.exists() else []
    if not runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation report exists yet. Run evaluation/run.py first.",
        )
    return runs[0]


@router.get("/latest")
def latest_evaluation(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return the newest aggregate report when explicitly enabled locally."""
    if not settings.evaluation_dashboard_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    report_directory = _latest_report_directory()
    try:
        metrics = json.loads((report_directory / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The latest evaluation report could not be read.",
        ) from exc
    return {"run_id": report_directory.name, "metrics": metrics}
