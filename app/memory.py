import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional


DB_PATH = Path(os.getenv("CHAT_HISTORY_DB_PATH", "Data/chat_history.db"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_chat_history_db() -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def store_chat_message(session_id: str, role: str, content: str) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content),
        )
        connection.commit()


def fetch_chat_history(session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    with closing(_connect()) as connection:
        if limit is None:
            cursor = connection.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
        else:
            cursor = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, id
                    FROM chat_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            )
        rows = cursor.fetchall()

    return [{"role": str(role), "content": str(content)} for role, content in rows]


def clear_chat_history(session_id: str) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            DELETE FROM chat_history
            WHERE session_id = ?
            """,
            (session_id,),
        )
        connection.commit()
