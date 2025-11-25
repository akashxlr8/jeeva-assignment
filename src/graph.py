from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, START, END, StateGraph
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import sqlite3
from dotenv import load_dotenv
from .personas import persona_manager, PERSONAS

load_dotenv()

# Global variable for current user_id (Note: In a real async server, this global is not thread-safe. 
# We should rely on config['configurable']['user_id'] inside nodes, but tools might need context.
# For this assignment, we'll stick to the pattern but be aware of limitations.)
current_user_id = None

# Create checkpointer and store
# check_same_thread=False is recommended for multi-threaded environments (like web apps)
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Dedicated SQLite store to persist procedural/user memory across runs
store_conn = sqlite3.connect("store.sqlite", check_same_thread=False, isolation_level=None)
store = SqliteStore(store_conn)
store.setup()

# Define tools
# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers."""
#     return a * b

# @tool
# def add(a: int, b: int) -> int:
#     """Add two numbers."""
#     return a + b

# @tool
# def save_user_info(user_info: str) -> str:
#     """Save user info to long-term memory."""
#     global current_user_id
#     if not current_user_id:
#         return "Error: No user context."
        
#     # Parse user_info
#     info_dict = {}
#     try:
#         for item in user_info.split(', '):
#             if '=' in item:
#                 key, value = item.split('=', 1)
#                 if value.isdigit():
#                     value = int(value)
#                 info_dict[key] = value
#         store.put(("users",), current_user_id, info_dict)
#         return "Successfully saved user info."
#     except Exception as e:
#         return f"Error saving info: {str(e)}"

# @tool
# def get_user_info() -> str:
#     """Retrieve user info from long-term memory."""
#     global current_user_id
#     if not current_user_id:
#         return "Error: No user context."
        
#     user_info = store.get(("users",), current_user_id)
#     return str(user_info.value) if user_info else "No user profile found."

# @tool
# def update_instructions(new_instructions: str) -> str:
#     """Update the agent's system instructions for procedural memory."""
#     store.put(("procedural",), "instructions", {"content": new_instructions})
#     return "Instructions updated successfully."

# Create model with tools
model = init_chat_model(
    "gpt-4.1-mini", 
    temperature=0,
    max_tokens=1000
)
tools = []
# tools = [multiply, add, save_user_info, get_user_info, update_instructions]
llm_with_tools = model.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

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
    # Map persona name to prompt key
    prompt_key = persona_name.lower()
    if prompt_key not in PERSONAS:
        prompt_key = "base"
        
    base_system = PERSONAS.get(prompt_key, PERSONAS["base"])
    
    # Retrieve procedural memory (instructions)
    instructions = store.get(("procedural",), "instructions")
    system_content = instructions.value["content"] if instructions else base_system
    
    # Retrieve user profile for semantic memory
    profile = store.get(("users",), user_id)
    profile_str = str(profile.value) if profile else "No profile available"
    
    system_content += f"\n\nUser profile: {profile_str}"
    system_content += f"\n\nCurrent Persona: {persona_name}"
    system_content += "\n\nNote: Use the conversation history to answer questions about previous interactions. Only use tools if you need to perform a specific action or retrieve data not present in the chat."
    
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
    # Ensure global context is set for tools
    global current_user_id
    current_user_id = config["configurable"].get("user_id", "unknown")

    result = []
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls'):
        for tool_call in last_msg.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    return END


# Build workflow
workflow = StateGraph(MessagesState)

workflow.add_node("llm_call", llm_call)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "llm_call")
workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
workflow.add_edge("tool_node", "llm_call")

# Compile the agent
graph = workflow.compile(checkpointer=checkpointer, store=store)