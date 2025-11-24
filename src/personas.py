from typing import Dict

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
    if "act like my mentor" in message_lower or "be my mentor" in message_lower or "now act like my mentor" in message_lower:
        return "mentor"
    elif "act like an investor" in message_lower or "be an investor" in message_lower or "now act like an investor" in message_lower:
        return "investor"
    # Add more detection logic
    return "base"