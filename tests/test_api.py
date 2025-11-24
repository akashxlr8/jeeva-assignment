import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat", json={"user_id": "test_user", "message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "thread" in data

def test_chat_history_endpoint():
    response = client.get("/chat_history?user_id=test_user")
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "history" in data