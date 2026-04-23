import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

def get_checkpointer():
    """
    Returns a SQLite checkpointer for persistent memory.
    Saves all graph state to a local database file.
    """
    os.makedirs("memory", exist_ok=True)
    
    db_path = "memory/research_agent.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    print(f"[MEMORY] Checkpointer initialized at: {db_path}")
    return checkpointer