from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, START, END, StateGraph
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    
from dotenv import load_dotenv
from dataclasses import dataclass
load_dotenv()

# Global variable for current user_id
current_user_id = None

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

# Create model with tools
model = init_chat_model(
    "gpt-4.1-mini",
    temperature=0
)
tools = [multiply, add, save_user_info, get_user_info]
llm_with_tools = model.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

# Nodes
def llm_call(state: MessagesState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    global current_user_id
    user_id = config["configurable"].get("user_id", "unknown")
    current_user_id = user_id
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
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


# Invoke with thread_id AND user_id
user_id = "user_123"
config = RunnableConfig(configurable={
    "thread_id": "conversation_1",  # For checkpoints
    "user_id": user_id              # For memory store namespace
})

result = graph.invoke({"messages": [HumanMessage(content="What is 5 * 3?")]}, config)

for m in result["messages"]:
    m.pretty_print()

# Demonstrate agent calling memory tools (save then retrieve)
save_msg = [HumanMessage(content="Save my profile: name=Alice, age=30")]
res_save = graph.invoke({"messages": save_msg}, config)
print("-- after save attempt --")
for m in res_save["messages"]:
    m.pretty_print()

get_msg = [HumanMessage(content="Get my profile")]
res_get = graph.invoke({"messages": get_msg}, config)
print("-- after get attempt --")
for m in res_get["messages"]:
    m.pretty_print()

# Example: Save and retrieve long-term memory for user
user_namespace = (user_id, "profile")
store.put(user_namespace, "user_profile", {"name": "Akash", "preferences": ["SaaS", "AI"]})
profile = store.get(user_namespace, "user_profile")
print("User profile:", profile.value if profile else None)

# Example: Search for memories in namespace
items = store.search(user_namespace, filter={"preferences": "AI"}, query="preferences")
print("Search results:", [item.value for item in items])

