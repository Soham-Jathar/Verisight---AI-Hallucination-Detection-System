from app.services.documents import DOCUMENT_CHUNK_OVERLAP_WORDS, DOCUMENT_CHUNK_WORDS, _document_chunks


def test_document_chunks_overlap_without_dropping_text() -> None:
    words = [f"word{index}" for index in range(DOCUMENT_CHUNK_WORDS + 80)]
    chunks = _document_chunks(" ".join(words))

    assert len(chunks) == 2
    assert chunks[0].split()[-DOCUMENT_CHUNK_OVERLAP_WORDS:] == chunks[1].split()[:DOCUMENT_CHUNK_OVERLAP_WORDS]
    assert words[-1] in chunks[-1]
