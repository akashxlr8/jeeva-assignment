from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from .graph import graph, store
from .personas import persona_manager, detect_persona_request
import uuid

app = FastAPI(title="Persona-Switching Chatbot")

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatHistoryRequest(BaseModel):
    user_id: str

def get_user_threads(user_id: str):
    """Retrieve user's thread mapping from the store."""
    item = store.get(("config",), f"threads_{user_id}")
    return item.value if item else {}

def save_user_threads(user_id: str, threads: dict):
    """Save user's thread mapping to the store."""
    store.put(("config",), f"threads_{user_id}", threads)

@app.post("/chat")
def chat(request: ChatRequest):
    user_id = request.user_id
    message = request.message

    # 1. Load persistent thread mapping
    user_threads = get_user_threads(user_id)
    # Sync with in-memory manager (optional, but good for consistency if we used it elsewhere)
    persona_manager.user_threads[user_id] = user_threads

    # 2. Router Logic
    target_persona = detect_persona_request(message)
    
    # Determine active thread
    active_thread_item = store.get(("config",), f"active_thread_{user_id}")
    active_thread_id = active_thread_item.value if active_thread_item else None
    
    thread_id = None
    persona_name = "Business Domain Expert" # Default

    if target_persona != "base":
        # Switching to specific persona
        persona_name = target_persona.capitalize() # e.g. "Mentor"
        if persona_name in user_threads:
            thread_id = user_threads[persona_name]
        else:
            thread_id = str(uuid.uuid4())
            user_threads[persona_name] = thread_id
            save_user_threads(user_id, user_threads)
            
        # Set as active
        store.put(("config",), f"active_thread_{user_id}", thread_id)
        persona_manager.thread_personas[thread_id] = persona_name
        
    else:
        # Continue active thread
        if active_thread_id:
            thread_id = active_thread_id
            # Find persona name
            for p, t in user_threads.items():
                if t == thread_id:
                    persona_name = p
                    break
            persona_manager.thread_personas[thread_id] = persona_name
        else:
            # No active thread, start default
            persona_name = "Business Domain Expert"
            if persona_name in user_threads:
                thread_id = user_threads[persona_name]
            else:
                thread_id = str(uuid.uuid4())
                user_threads[persona_name] = thread_id
                save_user_threads(user_id, user_threads)
            
            store.put(("config",), f"active_thread_{user_id}", thread_id)
            persona_manager.thread_personas[thread_id] = persona_name

    # Ensure persona_manager has the mapping
    persona_manager.thread_personas[thread_id] = persona_name

    # 3. Invoke Graph
    config = RunnableConfig(configurable={
        "thread_id": thread_id,
        "user_id": user_id
    })

    try:
        # Invoke
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
        
        # Get last AI message
        last_msg = result["messages"][-1]
        response_content = last_msg.content if last_msg.type == "ai" else "..."
        
        return {
            "response": response_content,
            "thread_id": thread_id,
            "persona": persona_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history")
def get_chat_history(user_id: str):
    user_threads = get_user_threads(user_id)
    history = {}
    
    for persona, thread_id in user_threads.items():
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state and state.values:
            messages = state.values.get("messages", [])
            history[persona] = [
                {"role": msg.type, "content": msg.content}
                for msg in messages
            ]
            
    return {"user_id": user_id, "history": history}