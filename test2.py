from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing import TypedDict
    
from dotenv import load_dotenv
from dataclasses import dataclass
load_dotenv()

# Create checkpointer and store
checkpointer = InMemorySaver()
store = InMemoryStore()

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
def save_user_info(user_info: dict, runtime):
    """Save user info to long-term memory."""
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

# Example: Add a tool to retrieve user info from long-term memory
@tool
def get_user_info(runtime):
    """Retrieve user info from long-term memory."""
    store = runtime.store
    user_id = runtime.context.user_id
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Create agent
agent = create_agent(
    model=ChatOpenAI(),
    tools=[multiply, add, save_user_info, get_user_info],
    checkpointer=checkpointer,
    store=store,
)

# Depending on LangChain version, `create_agent` may return a compiled/ready-to-invoke
# object. Use `agent` directly as the runnable/graph.
graph = agent

# Print registered tool names for verification
try:
    names = []
    for t in getattr(agent, "tools", []):
        if hasattr(t, "name"):
            names.append(getattr(t, "name"))
        elif callable(t):
            names.append(getattr(t, "__name__", repr(t)))
        else:
            names.append(repr(t))
    print("Registered agent tools:", names)
except Exception:
    print("Registered agent tools: unable to inspect")

# Invoke with thread_id AND user_id
user_id = "user_123"
config = RunnableConfig(configurable={
    "thread_id": "conversation_1",  # For checkpoints
    "user_id": user_id              # For memory store namespace
})

@dataclass
class Context:
    user_id: str

result = graph.invoke({"messages": [{"role": "user", "content": "What is 5 * 3?"}]}, config, context=Context(user_id=user_id))


def print_agent_result(res):
    """Robust printer for different agent result shapes."""
    print("=== Agent invoke result ===")
    try:
        if isinstance(res, dict) and "messages" in res:
            msgs = res["messages"]
        elif hasattr(res, "messages"):
            msgs = getattr(res, "messages")
        else:
            # Fallback: print raw object/value
            print(res)
            return

        for idx, m in enumerate(msgs, start=1):
            # Determine message type name
            mtype = type(m).__name__ if not isinstance(m, dict) else "dict"

            # Try to extract a role-like label
            role = None
            if hasattr(m, "role"):
                role = getattr(m, "role")
            elif hasattr(m, "type"):
                role = getattr(m, "type")
            elif isinstance(m, dict):
                role = m.get("role") or m.get("type")

            # Try to extract textual content
            content = None
            if hasattr(m, "content"):
                content = getattr(m, "content")
            elif hasattr(m, "text"):
                content = getattr(m, "text")
            elif isinstance(m, dict):
                # Many messages are simple dicts with role/content
                content = m.get("content") or m.get("text") or repr(m)

            # Ensure we always display something useful
            role_label = role if role is not None else f"<{mtype}>"
            content_label = content if content is not None else "<no content>"
            print(f"[{idx}] {role_label}: {content_label}")
    except Exception as e:
        print("Error while printing agent result:", e)


print_agent_result(result)

# Demonstrate agent calling memory tools (save then retrieve)
save_msg = {"messages": [{"role": "user", "content": "Save my profile: name=Alice, age=30"}]} 
res_save = graph.invoke(save_msg, config, context=Context(user_id=user_id))
print("-- after save attempt --")
print_agent_result(res_save)

get_msg = {"messages": [{"role": "user", "content": "Get my profile"}]}
res_get = graph.invoke(get_msg, config, context=Context(user_id=user_id))
print("-- after get attempt --")
print_agent_result(res_get)

# Example: Save and retrieve long-term memory for user
user_namespace = (user_id, "profile")
store.put(user_namespace, "user_profile", {"name": "Akash", "preferences": ["SaaS", "AI"]})
profile = store.get(user_namespace, "user_profile")
print("User profile:", profile.value if profile else None)

# Example: Search for memories in namespace
items = store.search(user_namespace, filter={"preferences": "AI"}, query="preferences")
print("Search results:", [item.value for item in items])

# Example: Add a tool to update user info in long-term memory
# ...existing tools defined earlier (save_user_info, get_user_info) are used here; no duplicates.

# Example: Trim messages to fit context window
def trim_messages(messages, max_messages=3):
    if len(messages) <= max_messages:
        return messages
    return [messages[0]] + messages[-max_messages:]

# Example usage of trim_messages
trimmed = trim_messages([
    {"role": "user", "content": "hi! I'm bob"},
    {"role": "ai", "content": "Hi Bob!"},
    {"role": "user", "content": "what's my name?"},
    {"role": "ai", "content": "Your name is Bob."}
])
print("Trimmed messages:", trimmed)
