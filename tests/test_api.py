"""
Tests for the Persona-Switching Agentic Chatbot API.   

"""

import uuid
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_personas_endpoint_lists_defaults():
    response = client.get("/personas")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("personas"), list)
    for expected in ("base", "mentor", "investor"):
        assert expected in data["personas"], f"Expected persona '{expected}' in {data['personas']}"


def test_chat_persona_switching_and_history():
    user_id = f"test_user_{uuid.uuid4().hex}"

    base_resp = client.post("/chat", json={"user_id": user_id, "message": "Hello, who are you?"})
    assert base_resp.status_code == 200
    base_data = base_resp.json()
    assert base_data["persona"] == "Business Domain Expert"

    mentor_resp = client.post(
        "/chat",
        json={"user_id": user_id, "message": "Act like my mentor. How can I improve?"},
    )
    assert mentor_resp.status_code == 200
    mentor_data = mentor_resp.json()
    assert mentor_data["persona"] == "Mentor"
    assert mentor_data["thread_id"] != base_data["thread_id"]

    investor_resp = client.post(
        "/chat",
        json={"user_id": user_id, "message": "Switch to investor. What is the ROI?"},
    )
    assert investor_resp.status_code == 200
    investor_data = investor_resp.json()
    assert investor_data["persona"] == "Investor"
    assert investor_data["thread_id"] != mentor_data["thread_id"]

    history_resp = client.get(f"/chat_history?user_id={user_id}")
    assert history_resp.status_code == 200
    history = history_resp.json().get("history", {})
    assert "Business Domain Expert" in history
    assert "Mentor" in history
    assert "Investor" in history
    for persona, entries in history.items():
        assert entries, f"Expected history entries for {persona}"


def test_chat_with_empty_message():
    user_id = f"test_user_{uuid.uuid4().hex}"
    response = client.post("/chat", json={"user_id": user_id, "message": ""})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "persona" in data


def test_chat_history_empty():
    user_id = f"test_user_{uuid.uuid4().hex}"
    response = client.get(f"/chat_history?user_id={user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user_id
    assert data["history"] == {}


def test_multiple_messages_same_thread():
    user_id = f"test_user_{uuid.uuid4().hex}"
    
    # Start mentor thread
    resp1 = client.post("/chat", json={"user_id": user_id, "message": "Act like my mentor."})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["persona"] == "Mentor"
    thread_id = data1["thread_id"]
    
    # Send another message in same thread
    resp2 = client.post("/chat", json={"user_id": user_id, "message": "How can I improve my skills?"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["persona"] == "Mentor"
    assert data2["thread_id"] == thread_id  # Same thread
    
    # Check history
    history_resp = client.get(f"/chat_history?user_id={user_id}")
    assert history_resp.status_code == 200
    history = history_resp.json()["history"]
    assert "Mentor" in history
    assert len(history["Mentor"]) == 4  # Two exchanges: human1, ai1, human2, ai2


def test_switch_back_to_previous_persona():
    user_id = f"test_user_{uuid.uuid4().hex}"
    
    # Mentor
    mentor_resp = client.post("/chat", json={"user_id": user_id, "message": "Be my mentor."})
    assert mentor_resp.status_code == 200
    mentor_data = mentor_resp.json()
    assert mentor_data["persona"] == "Mentor"
    mentor_thread = mentor_data["thread_id"]
    
    # Investor
    investor_resp = client.post("/chat", json={"user_id": user_id, "message": "Now act like an investor."})
    assert investor_resp.status_code == 200
    investor_data = investor_resp.json()
    assert investor_data["persona"] == "Investor"
    investor_thread = investor_data["thread_id"]
    assert investor_thread != mentor_thread
    
    # Back to mentor
    back_resp = client.post("/chat", json={"user_id": user_id, "message": "Back to mentor. What did we talk about?"})
    assert back_resp.status_code == 200
    back_data = back_resp.json()
    assert back_data["persona"] == "Mentor"
    assert back_data["thread_id"] == mentor_thread  # Back to original mentor thread


def test_persona_creation():
    user_id = f"test_user_{uuid.uuid4().hex}"
    
    # Create a new persona
    resp = client.post("/chat", json={"user_id": user_id, "message": "Be a pirate. Arrr!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["persona"] == "Pirate"  # Assuming it creates "pirate" and capitalizes
    
    # Check if it's now in personas list
    personas_resp = client.get("/personas")
    assert personas_resp.status_code == 200
    personas = personas_resp.json()["personas"]
    assert "pirate" in personas


def test_chat_missing_user_id():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 422  # Validation error


def test_chat_history_missing_user_id():
    response = client.get("/chat_history")
    assert response.status_code == 422  # Validation error