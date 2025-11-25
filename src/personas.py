import uuid
import sqlite3
from typing import Dict, Optional, Literal, get_args
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI

DB_PATH = "personas.db"

# Define default personas for seeding
DEFAULT_PERSONAS: Dict[str, str] = {
    "base": "You are a Business Domain Expert capable of assuming various professional roles. Adapt your responses based on the requested persona.",
    "mentor": "You are an experienced mentor guiding the user on business and product development. Provide thoughtful advice, ask probing questions, and draw from real-world examples.",
    "investor": "You are a skeptical investor evaluating business opportunities. Focus on metrics like TAM, SAM, SOM, revenue models, and risks. Be critical and data-driven.",
}

def init_personas_db():
    """Initialize the personas database and seed with defaults if empty."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS personas (
            name TEXT PRIMARY KEY,
            prompt TEXT
        )
    ''')
    
    cursor.execute('SELECT count(*) FROM personas')
    if cursor.fetchone()[0] == 0:
        print("Seeding personas DB with defaults...")
        for name, prompt in DEFAULT_PERSONAS.items():
            cursor.execute('INSERT INTO personas (name, prompt) VALUES (?, ?)', (name, prompt))
        conn.commit()
    conn.close()

def load_personas() -> Dict[str, str]:
    """Load all personas from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, prompt FROM personas')
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def save_persona_to_db(name: str, prompt: str):
    """Save a new persona to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO personas (name, prompt) VALUES (?, ?)', (name, prompt))
    conn.commit()
    conn.close()

# Initialize DB and load personas into memory
init_personas_db()
PERSONAS: Dict[str, str] = load_personas()

class PersonaDecision(BaseModel):
    """Decision on whether to switch persona, create a new one, or continue."""
    thinking: str = Field(description="Reasoning and thinking for the decision.")
    action: Literal["switch", "create", "continue"] = Field(
        description="The action to take. 'switch' for existing personas, 'create' for new ones, 'continue' to stay."
    )
    target_persona: Optional[str] = Field(
        default=None,
        description="The existing persona to switch to. Required if action is 'switch'."
    )
    new_persona_name: Optional[str] = Field(
        default=None,
        description="Name of the new persona. Required if action is 'create'."
    )
    new_persona_description: Optional[str] = Field(
        default=None,
        description="Description of the new persona. Required if action is 'create'."
    )

    @model_validator(mode='after')
    def validate_decision(self):
        if self.action == 'switch':
            if not self.target_persona:
                raise ValueError("target_persona is required for switch action")
            if self.target_persona not in PERSONAS:
                raise ValueError(f"Invalid persona: {self.target_persona}. Must be one of {list(PERSONAS.keys())}")
        return self

def generate_new_persona_prompt(name: str, description: str) -> str:
    """Generate a system prompt for a new persona using an LLM."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    prompt = f"""Generate a system prompt for a persona named '{name}'.
    Description: {description}
    
    The prompt should start with "You are {name}..." and define the tone, style, and expertise.
    Keep it concise (2-3 sentences).
    """
    response = llm.invoke(prompt)
    return str(response.content)

def detect_persona_request(message: str) -> str:
    """Detect intent, handle persona creation if needed, and return the target persona name."""
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    structured_llm = llm.with_structured_output(PersonaDecision)
    
    available_personas = ", ".join(PERSONAS.keys())
    
    system_prompt = f"""You are an intent classifier for a persona-switching chatbot.
    Available personas: {available_personas}

    Return a structured object that matches the `PersonaDecision` model exactly.
    The object must be JSON-like with these fields:
        - thinking: short reasoning string
        - action: one of "switch", "create", "continue"
        - target_persona: (when action is "switch") existing persona name
        - new_persona_name: (when action is "create") name for the new persona
        - new_persona_description: (when action is "create") short description for the new persona

    Few-shot examples (exact structured output expected):

    User: "Act like a mentor, help me with constant burnouts"
    Output:
    {{
        "thinking": "User explicitly asked to act as a mentor and requested guidance",
        "action": "switch",
        "target_persona": "mentor"
    }}

    User: "Be a pirate"
    Output:
    {{
        "thinking": "User explicitly requested a pirate persona not in the list",
        "action": "create",
        "new_persona_name": "pirate",
        "new_persona_description": "A swashbuckling pirate persona: uses nautical metaphors, bold informal tone, and adventurous examples."
    }}

    User: "How do I scale my product?"
    Output:
    {{
        "thinking": "User asks a general product question without requesting a persona change",
        "action": "continue"
    }}

    Important: ONLY return the structured object (no extra commentary). Use persona names as lowercase keys that match the available persona list when switching. If creating, return a concise description (1-2 sentences).
    """

    try:
        decision = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ])
        
        # Handle potential dict return from structured_llm
        if isinstance(decision, dict):
            action = decision.get("action")
            target_persona = decision.get("target_persona")
            new_persona_name = decision.get("new_persona_name")
            new_persona_description = decision.get("new_persona_description")
        else:
            action = getattr(decision, "action", None)
            target_persona = getattr(decision, "target_persona", None)
            new_persona_name = getattr(decision, "new_persona_name", None)
            new_persona_description = getattr(decision, "new_persona_description", None)
        
        if action == "switch":
            return str(target_persona) if target_persona else "base"
            
        elif action == "create":
            name = new_persona_name.lower() if new_persona_name else "unknown"
            if name in PERSONAS:
                return name
            
            description = new_persona_description or f"A {name} persona."
            print(f"Creating new persona: {name}")
            new_prompt = generate_new_persona_prompt(name, description)
            PERSONAS[name] = new_prompt
            save_persona_to_db(name, new_prompt)
            return name
            
        else: # continue
            return "base"
            
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