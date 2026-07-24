from __future__ import annotations

import io
import re
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from pypdf import PdfReader

from app.schemas import DocumentInfo, EvidenceSource

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_CHARACTERS = 100_000
_documents: dict[str, "StoredDocument"] = {}


@dataclass
class StoredDocument:
    info: DocumentInfo
    text: str


async def save_pdf(upload: UploadFile) -> DocumentInfo:
    if upload.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a PDF file.")

    content = await upload.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF files must be 10 MB or smaller.")

    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This PDF could not be read.") from exc

    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No selectable text was found in this PDF. Scanned PDFs need OCR, which is not enabled yet.",
        )

    text = text[:MAX_DOCUMENT_CHARACTERS]
    document_id = uuid4().hex
    info = DocumentInfo(
        id=document_id,
        filename=upload.filename or "uploaded-document.pdf",
        pages=len(reader.pages),
        characters=len(text),
    )
    _documents[document_id] = StoredDocument(info=info, text=text)
    return info


def document_evidence(document_id: str, question: str, *, limit: int = 4) -> list[EvidenceSource]:
    document = _documents.get(document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The uploaded PDF is no longer available. Please upload it again.")

    terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+|\n+", document.text)
    ranked = sorted(
        ((len(terms.intersection(set(re.findall(r"[a-zA-Z0-9]{3,}", sentence.lower())))), sentence.strip()) for sentence in sentences if sentence.strip()),
        key=lambda item: item[0],
        reverse=True,
    )
    snippets = [sentence for _, sentence in ranked[:limit] if sentence] or [document.text[:800]]
    return [
        EvidenceSource(title=document.info.filename, url=f"document://{document.info.id}", snippet=snippet[:900])
        for snippet in snippets
    ]
