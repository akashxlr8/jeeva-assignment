import uuid
from typing import Dict, Optional

# Define persona system prompts
PERSONAS: Dict[str, str] = {
    "base": "You are a Business Domain Expert capable of assuming various professional roles. Adapt your responses based on the requested persona.",
    "mentor": "You are an experienced mentor guiding the user on business and product development. Provide thoughtful advice, ask probing questions, and draw from real-world examples.",
    "investor": "You are a skeptical investor evaluating business opportunities. Focus on metrics like TAM, SAM, SOM, revenue models, and risks. Be critical and data-driven.",
    # Add more personas as needed
}

def get_persona_prompt(persona: str) -> str:
    """Get the system prompt for a given persona."""
    return PERSONAS.get(persona.lower(), PERSONAS["base"])

def detect_persona_request(message: str) -> str:
    """Detect if the user is requesting a new persona and return the persona name."""
    message_lower = message.lower()
    
    # Mentor triggers
    if any(phrase in message_lower for phrase in [
        "act like my mentor", "be my mentor", "switch to mentor", "back to mentor", "act as a mentor"
    ]):
        return "mentor"
        
    # Investor triggers
    elif any(phrase in message_lower for phrase in [
        "act like an investor", "be an investor", "switch to investor", "back to investor", "act as an investor"
    ]):
        return "investor"
        
    # Add more detection logic
    return "base"

class PersonaManager:
    def __init__(self):
        # user_id -> {persona_name: thread_id}
        self.user_threads = {}
        # user_id -> active_thread_id
        self.active_threads = {}
        # thread_id -> persona_name
        self.thread_personas = {}

    def get_or_create_thread(self, user_id: str, persona_name: str) -> str:
        if user_id not in self.user_threads:
            self.user_threads[user_id] = {}
        
        if persona_name not in self.user_threads[user_id]:
            # Create new thread ID
            thread_id = str(uuid.uuid4())
            self.user_threads[user_id][persona_name] = thread_id
            self.thread_personas[thread_id] = persona_name
            print(f"[PersonaManager] Created new thread {thread_id} for persona '{persona_name}'")
        
        return self.user_threads[user_id][persona_name]

    def set_active_thread(self, user_id: str, thread_id: str):
        self.active_threads[user_id] = thread_id
        persona = self.thread_personas.get(thread_id, "Unknown")
        print(f"[PersonaManager] Switched active thread to {thread_id} (Persona: {persona})")

    def get_active_thread(self, user_id: str) -> Optional[str]:
        return self.active_threads.get(user_id)

    def get_persona_by_thread(self, thread_id: str) -> str:
        return self.thread_personas.get(thread_id, "base")

# Initialize global manager
persona_manager = PersonaManager()