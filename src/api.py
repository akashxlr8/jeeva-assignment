from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_message_histories import SQLChatMessageHistory
from .graph import runnable
from .personas import detect_persona_request, get_persona_prompt
import sqlite3

app = FastAPI(title="Persona-Switching Chatbot")

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatHistoryRequest(BaseModel):
    user_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    user_id = request.user_id
    message = request.message

    # Detect persona
    persona = detect_persona_request(message)
    if persona == "base":
        # Check for switch commands
        message_lower = message.lower()
        if "back to" in message_lower:
            if "mentor" in message_lower:
                persona = "mentor"
            elif "investor" in message_lower:
                persona = "investor"
        # Else, use default or last, but for simplicity, base

    thread_name = persona if persona != "base" else "default"
    session_id = f"{user_id}_{thread_name}"
    system_prompt = get_persona_prompt(persona)

    try:
        response = runnable.invoke(
            {"input": message, "system_prompt": system_prompt},
            config={"configurable": {"session_id": session_id}}
        )
        return {"response": response.content, "thread": thread_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history")
def get_chat_history(user_id: str):
    # Connect to db
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM message_store WHERE session_id LIKE ?", (f"{user_id}_%",))
    sessions = cursor.fetchall()
    history = {}
    for (session_id,) in sessions:
        thread_name = session_id.split("_", 1)[1] if "_" in session_id else session_id
        hist = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat.db")
        messages = hist.messages
        history[thread_name] = [
            {"role": "human" if msg.__class__.__name__ == "HumanMessage" else "ai", "content": msg.content}
            for msg in messages
        ]
    conn.close()
    return {"user_id": user_id, "history": history}