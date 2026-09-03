from app.services.documents import (
    DOCUMENT_CHUNK_OVERLAP_WORDS,
    DOCUMENT_CHUNK_WORDS,
    _document_chunks,
    _secondary_paper_options_from_text,
)


def test_document_chunks_overlap_without_dropping_text() -> None:
    words = [f"word{index}" for index in range(DOCUMENT_CHUNK_WORDS + 80)]
    chunks = _document_chunks(" ".join(words))

    assert len(chunks) == 2
    assert chunks[0].split()[-DOCUMENT_CHUNK_OVERLAP_WORDS:] == chunks[1].split()[:DOCUMENT_CHUNK_OVERLAP_WORDS]
    assert words[-1] in chunks[-1]


def test_secondary_paper_table_returns_every_requested_option() -> None:
    text = """Table 4: Allowed two test paper combinations in GATE 2027
CS DA, EC, GE, MA, PH, RA, ST, ME  PE CH
DA CS, EC, EE, MA, ME, PH, RA, ST, XE  PI ME, XE
6.4 Application Process"""

    answer = _secondary_paper_options_from_text(text, "What are the second paper options for GATE CS and DA?")

    assert answer == (
        "For CS, the second paper options are:\n"
        "1. DA\n2. EC\n3. GE\n4. MA\n5. PH\n6. RA\n7. ST\n8. ME\n\n"
        "For DA, the second paper options are:\n"
        "1. CS\n2. EC\n3. EE\n4. MA\n5. ME\n6. PH\n7. RA\n8. ST\n9. XE"
    )
