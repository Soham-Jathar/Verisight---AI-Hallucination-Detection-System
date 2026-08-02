import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EvidenceSource
from app.services.pipeline import _merge_evidence
from app.services.generator import _connection_error


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


def test_analyze_rejects_unsupported_mode(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        json={"question": "Who created Python?", "mode": "document"},
    )
    assert response.status_code == 501


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
