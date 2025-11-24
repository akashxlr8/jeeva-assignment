from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from .personas import get_persona_prompt, detect_persona_request

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

def get_history(session_id: str):
    """Get chat message history for a session."""
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat.db")

# Wrap with history
runnable = RunnableWithMessageHistory(
    chain,
    get_history,
    history_messages_key="history",
    input_messages_key="input"
)