"""Offline FastAPI API tests; no DeepSeek key or real API calls required."""

import pytest
from fastapi.testclient import TestClient

from main import create_app


class FakeService:
    def __init__(self):
        self._histories = {}

    def ask(self, thread_id: str, message: str):
        history = self._histories.setdefault(thread_id, [])
        history.append({"role": "user", "content": message})
        reply = f"测试回复：{message}"
        history.append({"role": "assistant", "content": reply})
        return {
            "thread_id": thread_id,
            "answer": reply,
            "status": "completed",
            "llm_calls": 1,
            "tool_calls": 0,
            "history_count": len(history),
        }

    def history(self, thread_id: str):
        return list(self._histories.get(thread_id, []))


@pytest.fixture
def client():
    app = create_app(service_factory=FakeService)
    # with 会运行 FastAPI lifespan，并在测试后自动清理。
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_success(client):
    response = client.post(
        "/chat", json={"thread_id": "study-a", "message": "你好"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "测试回复：你好"
    assert data["status"] == "completed"
    assert data["history_count"] == 2


def test_chat_invalid_input(client):
    for message in ["", "   "]:
        response = client.post(
            "/chat", json={"thread_id": "study-a", "message": message}
        )
        assert response.status_code == 422


def test_chat_invalid_thread_id(client):
    response = client.post(
        "/chat", json={"thread_id": "has spaces", "message": "你好"}
    )
    assert response.status_code == 422


def test_history_isolated_between_threads(client):
    client.post("/chat", json={"thread_id": "study-a", "message": "第一个问题"})
    client.post("/chat", json={"thread_id": "study-a", "message": "第二个问题"})
    client.post("/chat", json={"thread_id": "study-b", "message": "其他问题"})

    a = client.get("/threads/study-a/history").json()
    b = client.get("/threads/study-b/history").json()
    assert a["thread_id"] == "study-a"
    assert len(a["messages"]) == 4
    assert len(b["messages"]) == 2
    assert all("其他问题" not in msg["content"] for msg in a["messages"])
