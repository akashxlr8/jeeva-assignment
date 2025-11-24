from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def trim_conversation_messages(messages, max_tokens: int = 1000):
    """Trim messages to fit within token limit."""
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
    )