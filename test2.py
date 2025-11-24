from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, START, END, StateGraph
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import uuid
import sqlite3
    
from dotenv import load_dotenv
load_dotenv()

# Global variable for current user_id
current_user_id = None

# Create checkpointer and store
# check_same_thread=False is recommended for multi-threaded environments (like web apps)
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
store = InMemoryStore()

# --- Phase 1: Persona Management ---

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
        return self.thread_personas.get(thread_id, "Business Domain Expert")

# Initialize global manager
persona_manager = PersonaManager()

# System Prompts for Personas
SYSTEM_PROMPTS = {
    "Business Domain Expert": "You are a Business Domain Expert. You are capable of assuming more specific roles. Help the user with general business queries.",
    "Mentor": "You are a Mentor. You are supportive, encouraging, and focus on personal and professional growth. Draw on your experience to guide the user.",
    "Investor": "You are a Skeptical Investor. You are critical, focused on numbers, TAM, ROI, and scalability. Ask tough questions.",
}

# --- End Phase 1 Setup ---

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Example: Add a tool to update user info in long-term memory
@tool
def save_user_info(user_info: str) -> str:
    """Save user info to long-term memory."""
    global current_user_id
    # Parse user_info
    info_dict = {}
    for item in user_info.split(', '):
        key, value = item.split('=')
        if value.isdigit():
            value = int(value)
        info_dict[key] = value
    store.put(("users",), current_user_id, info_dict)
    return "Successfully saved user info."

# Example: Add a tool to retrieve user info from long-term memory
@tool
def get_user_info() -> str:
    """Retrieve user info from long-term memory."""
    global current_user_id
    user_info = store.get(("users",), current_user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Example: Add a tool to update agent's instructions (procedural memory)
@tool
def update_instructions(new_instructions: str) -> str:
    """Update the agent's system instructions for procedural memory."""
    store.put(("procedural",), "instructions", {"content": new_instructions})
    return "Instructions updated successfully."

# Create model with tools
model = init_chat_model(
    "gpt-4.1-mini",
    temperature=0
)
tools = [multiply, add, save_user_info, get_user_info, update_instructions]
llm_with_tools = model.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

# Example: Trim messages to fit context window
def trim_messages(messages, max_messages=10):
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]

# Nodes
def llm_call(state: MessagesState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    global current_user_id
    user_id = config["configurable"].get("user_id", "unknown")
    thread_id = config["configurable"].get("thread_id", "unknown")
    current_user_id = user_id
    
    # Determine Persona
    persona_name = persona_manager.get_persona_by_thread(thread_id)
    base_system = SYSTEM_PROMPTS.get(persona_name, SYSTEM_PROMPTS["Business Domain Expert"])
    
    # Retrieve procedural memory (instructions)
    instructions = store.get(("procedural",), "instructions")
    system_content = instructions.value["content"] if instructions else base_system
    
    # Retrieve user profile for semantic memory
    profile = store.get(("users",), user_id)
    profile_str = str(profile.value) if profile else "No profile available"
    
    system_content += f"\n\nUser profile: {profile_str}"
    system_content += f"\n\nCurrent Persona: {persona_name}"
    
    # Trim messages for short-term memory management
    trimmed_messages = trim_messages(state["messages"])
    
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(content=system_content)
                ]
                + trimmed_messages
            )
        ]
    }


def tool_node(state: MessagesState, config: RunnableConfig):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("llm_call", llm_call)
workflow.add_node("tool_node", tool_node)
# Add edges to connect nodes
workflow.add_edge(START, "llm_call")
workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
workflow.add_edge("tool_node", "llm_call")

# Compile the agent
graph = workflow.compile(checkpointer=checkpointer, store=store)


# --- Orchestrator Logic ---

def router(message: str) -> Optional[str]:
    """
    Determines if the user wants to switch to a specific persona.
    Returns the persona name if a switch is detected, else None.
    """
    # Simple keyword-based routing for Phase 1 (can be upgraded to LLM)
    msg_lower = message.lower()
    if "mentor" in msg_lower and ("act like" in msg_lower or "switch to" in msg_lower or "back to" in msg_lower):
        return "Mentor"
    if "investor" in msg_lower and ("act like" in msg_lower or "switch to" in msg_lower or "back to" in msg_lower):
        return "Investor"
    if "business" in msg_lower and ("act like" in msg_lower or "switch to" in msg_lower):
        return "Business Domain Expert"
    return None

def orchestrator(user_id: str, message: str):
    """
    Orchestrates the conversation:
    1. Checks for persona switch.
    2. Determines active thread.
    3. Invokes the graph.
    """
    # 1. Check for persona switch
    target_persona = router(message)
    
    if target_persona:
        # Switch detected: Get or create thread for this persona
        thread_id = persona_manager.get_or_create_thread(user_id, target_persona)
        persona_manager.set_active_thread(user_id, thread_id)
        print(f"--> Switching to persona: {target_persona}")
    else:
        # No switch: Use active thread, or default to Business Expert if none
        thread_id = persona_manager.get_active_thread(user_id)
        if not thread_id:
            # Default start
            target_persona = "Business Domain Expert"
            thread_id = persona_manager.get_or_create_thread(user_id, target_persona)
            persona_manager.set_active_thread(user_id, thread_id)
            print(f"--> Starting default persona: {target_persona}")

    # 2. Invoke Graph
    config = RunnableConfig(configurable={
        "thread_id": thread_id,
        "user_id": user_id
    })
    
    print(f"--> Invoking thread: {thread_id}")
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    
    # Print last AI response
    for m in reversed(result["messages"]):
        if m.type == "ai":
            print(f"AI ({persona_manager.get_persona_by_thread(thread_id)}): {m.content}")
            break

# --- Test Workflow ---

if __name__ == "__main__":
    user_id = "test_user_1"
    
    print("\n=== Step 1: Start as Mentor ===")
    orchestrator(user_id, "Act like my mentor. How can I scale my product?")
    
    print("\n=== Step 2: Continue as Mentor ===")
    orchestrator(user_id, "What are the first steps?")
    
    print("\n=== Step 3: Switch to Investor ===")
    orchestrator(user_id, "Now switch to investor. What is the TAM?")
    
    print("\n=== Step 4: Switch back to Mentor ===")
    orchestrator(user_id, "Back to mentor. What was I asking about scaling?")

