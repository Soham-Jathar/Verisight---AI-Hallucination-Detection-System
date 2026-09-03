import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ClaimAssessment, EvidenceSource
from app.services.pipeline import _claims_needing_focused_evidence, _merge_evidence
from app.services.generator import _connection_error, _document_grounding_instruction, _list_answer_instruction


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_analyze_requires_question(client: TestClient) -> None:
    response = client.post("/api/analyze", json={"question": "ab", "mode": "web"})
    assert response.status_code == 422


def test_merge_evidence_keeps_focused_excerpt_for_same_url() -> None:
    initial = EvidenceSource(
        title="Python",
        url="https://example.com/python",
        snippet="Python is a programming language.",
    )
    focused = EvidenceSource(
        title="Python",
        url="https://example.com/python",
        snippet="Python was created by Guido van Rossum and first released in 1991.",
    )

    merged = _merge_evidence([initial], [focused])

    assert len(merged) == 1
    assert "Guido van Rossum" in merged[0].snippet


def test_unresolved_claims_receive_focused_evidence_retrieval() -> None:
    claims = [
        ClaimAssessment(claim="Supported fact.", status="supported", confidence=0.99, rationale="ok"),
        ClaimAssessment(claim="Missing date.", status="uncertain", confidence=0.70, rationale="review"),
        ClaimAssessment(claim="Potentially contradicted event.", status="unsupported", confidence=0.80, rationale="check"),
    ]

    assert _claims_needing_focused_evidence(claims) == [
        "Missing date.",
        "Potentially contradicted event.",
    ]


def test_document_only_generation_requires_exact_document_values() -> None:
    document = EvidenceSource(
        title="syllabus.pdf",
        url="document://syllabus",
        snippet="GATE 2027 is organised by IIT Madras.",
    )

    instruction = _document_grounding_instruction([document])

    assert "authoritative" in instruction
    assert "exact value directly" in instruction


def test_factual_list_instruction_preserves_complete_supported_lists() -> None:
    instruction = _list_answer_instruction("Name all the GATE 2027 test papers")

    assert "every requested item" in instruction
    assert "six items" in instruction


def test_document_mode_requires_an_uploaded_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        json={"question": "Who created Python?", "mode": "document"},
    )
    assert response.status_code == 400
    assert "Upload a PDF" in response.json()["detail"]


def test_analyze_web_mode_contract(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        json={"question": "Who created Python?", "mode": "web"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "Who created Python?"
    assert payload["mode"] == "web"
    assert payload["stage"] == "complete"
    assert isinstance(payload["message"], str)
    assert payload["message"]


def test_empty_connection_error_keeps_a_useful_error_type() -> None:
    error = _connection_error("Gemini", httpx.ConnectError(""))

    assert "ConnectError" in str(error)
    assert "VPN, proxy, or firewall" in str(error)
