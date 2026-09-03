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
DOCUMENT_CHUNK_WORDS = 220
DOCUMENT_CHUNK_OVERLAP_WORDS = 40
DOCUMENT_QUERY_STOP_WORDS = {
    "about", "available", "does", "for", "from", "gate", "have", "how",
    "list", "name", "only", "options", "paper", "papers", "the", "there",
    "these", "what", "which", "with",
}
_documents: dict[str, "StoredDocument"] = {}


@dataclass
class StoredDocument:
    info: DocumentInfo
    text: str


def _document_chunks(text: str) -> list[str]:
    """Split a document into overlapping excerpts for local retrieval."""
    words = text.split()
    if len(words) <= DOCUMENT_CHUNK_WORDS:
        return [text]

    step = DOCUMENT_CHUNK_WORDS - DOCUMENT_CHUNK_OVERLAP_WORDS
    return [
        " ".join(words[start:start + DOCUMENT_CHUNK_WORDS])
        for start in range(0, len(words), step)
        if words[start:start + DOCUMENT_CHUNK_WORDS]
    ]


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

    # Exam and product documents frequently use two-character official codes
    # (for example CS, DA, AI, ML). Keeping them makes a lookup for a code
    # select the relevant table rather than a generic page containing "paper".
    terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]{2,}", question.lower())
        if term not in DOCUMENT_QUERY_STOP_WORDS
    }
    ranked = sorted(
        (
            (
                len(terms.intersection(set(re.findall(r"[a-zA-Z0-9]{3,}", chunk.lower())))),
                -index,
                chunk,
            )
            for index, chunk in enumerate(_document_chunks(document.text))
        ),
        reverse=True,
    )
    snippets = [chunk for _, _, chunk in ranked[:limit] if chunk] or [document.text[:1_500]]
    return [
        EvidenceSource(title=document.info.filename, url=f"document://{document.info.id}", snippet=snippet[:1_500])
        for snippet in snippets
    ]


def _secondary_paper_options_from_text(text: str, question: str) -> str | None:
    """Read an official primary→secondary code table without asking an LLM to infer rows."""
    requested_codes = re.findall(r"\b[A-Z]{2}\b", question.upper())
    if not requested_codes or not re.search(r"allowed two test paper combinations", text, re.IGNORECASE):
        return None

    table_match = re.search(
        r"Table 4: Allowed two test paper combinations.*?(?=\n\s*6\.4\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not table_match:
        return None

    options_by_primary: dict[str, list[str]] = {}
    for line in table_match.group(0).splitlines():
        # pypdf preserves the two visual columns as two or more spaces.
        for cell in re.split(r"\s{2,}", line.strip()):
            match = re.fullmatch(
                r"([A-Z]{2})\s+([A-Z]{2}(?:\s*,\s*[A-Z]{2})*|-)",
                cell.strip(),
            )
            if not match:
                continue
            primary, raw_options = match.groups()
            options_by_primary[primary] = [] if raw_options == "-" else re.findall(r"[A-Z]{2}", raw_options)

    selected = [code for code in requested_codes if code in options_by_primary]
    if not selected:
        return None

    sections: list[str] = []
    for primary in dict.fromkeys(selected):
        options = options_by_primary[primary]
        if not options:
            sections.append(f"For {primary}, no secondary paper option is listed.")
            continue
        items = "\n".join(f"{index}. {option}" for index, option in enumerate(options, start=1))
        sections.append(f"For {primary}, the second paper options are:\n{items}")
    return "\n\n".join(sections)


def document_secondary_paper_options(document_id: str, question: str) -> str | None:
    """Return all requested official second-paper codes when the uploaded document contains Table 4."""
    document = _documents.get(document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The uploaded PDF is no longer available. Please upload it again.")
    return _secondary_paper_options_from_text(document.text, question)
