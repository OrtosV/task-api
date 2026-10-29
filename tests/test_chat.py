from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_validates_empty_question() -> None:
    """Empty question must be rejected by Pydantic before reaching the chain."""
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_returns_answer_with_sources() -> None:
    """Smoke test with mocked chain and retriever."""
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "What is Ridge?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "Ridge uses L2" in body["answer"]
    assert len(body["sources"]) == 1
    assert "scikit-learn.org" in body["sources"][0]["url"]