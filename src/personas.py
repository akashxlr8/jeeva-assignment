import uuid
from typing import Dict, Optional, Literal, get_args
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define persona system prompts
PERSONAS: Dict[str, str] = {
    "base": "You are a Business Domain Expert capable of assuming various professional roles. Adapt your responses based on the requested persona.",
    "mentor": "You are an experienced mentor guiding the user on business and product development. Provide thoughtful advice, ask probing questions, and draw from real-world examples.",
    "investor": "You are a skeptical investor evaluating business opportunities. Focus on metrics like TAM, SAM, SOM, revenue models, and risks. Be critical and data-driven.",
    # Add more personas as needed
}

class PersonaDecision(BaseModel):
    """Decision on whether to switch persona or continue with the current one."""
    thinking: str = Field(description="Reasoning and thinking for the decision.")
    target_persona: str = Field(
        description="The persona the user wants to interact with. Use 'base' if no specific persona is requested or if the user wants to continue the current conversation without switching."
    )

    @classmethod
    def validate_target_persona(cls, v):
        if v not in PERSONAS:
            raise ValueError(f"Invalid persona: {v}. Must be one of {list(PERSONAS.keys())}")
        return v

def get_persona_prompt(persona: str) -> str:
    """Get the system prompt for a given persona."""
    return PERSONAS.get(persona.lower(), PERSONAS["base"])

def detect_persona_request(message: str) -> str:
    """Detect if the user is requesting a new persona and return the persona name using an LLM."""
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    structured_llm = llm.with_structured_output(PersonaDecision)
    
    # Dynamically generate available personas list (excluding 'base')
    available_personas = [k for k in PERSONAS.keys()]
    personas_list = ", ".join(available_personas)
    
    system_prompt = f"""You are an intent classifier for a persona-switching chatbot.
    Analyze the user's message to determine if they explicitly want to switch to a specific persona <persona list> ({personas_list}) </persona list> or if they are just continuing the conversation.
    
    - If the user says "act like my mentor", "switch to investor", "back to mentor", etc., return the corresponding persona.
    - If the user asks a question without specifying a persona change (e.g., "how do I scale?", "what is TAM?"), return 'base' to indicate no switch is requested (the system will handle context).
    - Only return a specific persona if the user EXPLICITLY requests a switch or context change.
    """
    
    try:
        decision = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ])
        return decision.target_persona
    except Exception as e:
        print(f"Error in persona detection: {e}")
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