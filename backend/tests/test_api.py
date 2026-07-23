import pytest
from fastapi.testclient import TestClient

from app.main import app


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
