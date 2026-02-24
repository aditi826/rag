"""
Database module for Supabase PostgreSQL connection.
Handles chat history and document metadata storage.
Falls back gracefully if database is unavailable.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

# Global flag to track if DB is available
_db_available = None


def get_connection():
    """Create and return a database connection."""
    global _db_available

    # If we already know DB is unavailable, don't retry
    if _db_available is False:
        return None

    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        _db_available = True
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        _db_available = False
        return None


def is_db_available() -> bool:
    """Check if database is available."""
    return _db_available is True


def init_database() -> bool:
    """Initialize database tables if they don't exist."""
    conn = get_connection()
    if conn is None:
        print("Database unavailable - chat history will be stored in session only")
        return False

    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title VARCHAR(500)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) REFERENCES chat_sessions(session_id),
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(500) NOT NULL,
                file_type VARCHAR(50),
                chunk_count INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database init error: {e}")
        return False


def create_session(session_id: str, title: Optional[str] = None):
    """Create a new chat session."""
    conn = get_connection()
    if conn is None:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (session_id, title) VALUES (%s, %s) ON CONFLICT (session_id) DO NOTHING",
            (session_id, title)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating session: {e}")


def save_message(session_id: str, role: str, content: str):
    """Save a chat message to the database."""
    conn = get_connection()
    if conn is None:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving message: {e}")


def get_chat_history(session_id: str, limit: int = 50) -> List[dict]:
    """Retrieve chat history for a session."""
    conn = get_connection()
    if conn is None:
        return []

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (session_id, limit)
        )
        messages = cur.fetchall()
        cur.close()
        conn.close()
        return messages
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []


def get_all_sessions() -> List[dict]:
    """Get all chat sessions."""
    conn = get_connection()
    if conn is None:
        return []

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT session_id, title, created_at
            FROM chat_sessions
            ORDER BY created_at DESC
            """
        )
        sessions = cur.fetchall()
        cur.close()
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error getting sessions: {e}")
        return []


def save_document(filename: str, file_type: str, chunk_count: int):
    """Save document metadata."""
    conn = get_connection()
    if conn is None:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, file_type, chunk_count) VALUES (%s, %s, %s)",
            (filename, file_type, chunk_count)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving document: {e}")


def get_documents() -> List[dict]:
    """Get all uploaded documents."""
    conn = get_connection()
    if conn is None:
        return []

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
        documents = cur.fetchall()
        cur.close()
        conn.close()
        return documents
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []
