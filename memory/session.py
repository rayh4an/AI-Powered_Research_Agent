import uuid
from datetime import datetime


def generate_thread_id() -> str:
    """Generates a unique thread ID for a new research session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"research_{timestamp}_{unique_id}"


def get_session_query(checkpointer, thread_id: str) -> str | None:
    """
    Looks up the original query from a saved session.
    Returns None if the session doesn't exist or has no query.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = checkpointer.get(config)
        if checkpoint:
            state = checkpoint.get("channel_values", {})
            return state.get("query", None)
        return None
    except Exception:
        return None


def load_session_history(checkpointer, thread_id: str) -> dict | None:
    """
    Loads the state of a previous session if it exists.
    Returns None if the session doesn't exist.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = checkpointer.get(config)
        if checkpoint:
            print(f"[MEMORY] Loaded existing session: {thread_id}")
            return checkpoint
        else:
            print(f"[MEMORY] No existing session found for: {thread_id}")
            return None
    except Exception as e:
        print(f"[MEMORY] Could not load session: {e}")
        return None


def list_all_sessions(checkpointer) -> list[str]:
    """Lists all saved research session thread IDs."""
    try:
        sessions = []
        for config, checkpoint, metadata in checkpointer.list({}):
            thread_id = config.get("configurable", {}).get("thread_id", "unknown")
            sessions.append(thread_id)
        return sessions
    except Exception as e:
        print(f"[MEMORY] Could not list sessions: {e}")
        return []