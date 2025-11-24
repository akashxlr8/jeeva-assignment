from langgraph.checkpoint.sqlite import SqliteSaver

def get_checkpointer(db_path: str = "chat.db"):
    """Get a SQLite checkpointer for persistent storage."""
    return SqliteSaver.from_conn_string(db_path)