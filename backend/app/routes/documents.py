from fastapi import APIRouter, File, UploadFile

from app.schemas import DocumentInfo
from app.services.documents import save_pdf

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Extract text from a PDF and retain it temporarily for this local session."""
    return await save_pdf(file)
