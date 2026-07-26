import os
import sqlite3
import json
import logging
from contextlib import closing
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("CHAT_HISTORY_DB_PATH", "Data/chat_history.db"))

# Rates: gpt-4o ($5/1M input, $15/1M output) and gpt-4o-mini ($0.15/1M input, $0.60/1M output)
MODEL_PRICING = {
    "gpt-4o": {"input_per_1m_tokens": 5.00, "output_per_1m_tokens": 15.00},
    "gpt-4o-mini": {"input_per_1m_tokens": 0.15, "output_per_1m_tokens": 0.60},
}


def log_event(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, default=str, ensure_ascii=True))

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_cost_db() -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                call_type TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()

def calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Fallback or default pricing
        return 0.0
    return (
        (max(input_tokens, 0) / 1_000_000.0) * pricing["input_per_1m_tokens"]
        + (max(output_tokens, 0) / 1_000_000.0) * pricing["output_per_1m_tokens"]
    )

def log_openai_cost(
    session_id: str,
    model: str,
    call_type: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    cost_usd = calculate_openai_cost(model, input_tokens, output_tokens)
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO llm_costs (session_id, model, call_type, input_tokens, output_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, model, call_type, input_tokens, output_tokens, cost_usd),
        )
        connection.commit()
    return cost_usd

def get_total_session_cost(session_id: str) -> float:
    with closing(_connect()) as connection:
        cursor = connection.execute(
            """
            SELECT COALESCE(SUM(cost_usd), 0)
            FROM llm_costs
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def clear_session_costs(session_id: str) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            DELETE FROM llm_costs
            WHERE session_id = ?
            """,
            (session_id,),
        )
        connection.commit()

def get_last_query_cost(session_id: str) -> dict:
    """Returns detailed breakdown of the last query cost."""
    with closing(_connect()) as connection:
        # We assume the last query costs are the most recent ones in the DB for this session
        # This is a bit simplified, but for a single-user session it works.
        # In a real app, we might want to group by a query_id.
        cursor = connection.execute(
            """
            SELECT call_type, cost_usd
            FROM llm_costs
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        
    breakdown = {"classifier": 0.0, "generator": 0.0, "total": 0.0}
    for call_type, cost in rows:
        if "intent_classification" in call_type:
            breakdown["classifier"] += cost
        elif "generation" in call_type:
            breakdown["generator"] += cost
        breakdown["total"] += cost
    return breakdown

def usage_tokens(usage: object, token_name: str) -> int:
    if usage is None:
        return 0
    value: Optional[int] = getattr(usage, token_name, None)
    if value is None and isinstance(usage, dict):
        raw_value = usage.get(token_name)
        value = int(raw_value) if raw_value is not None else 0
    return int(value or 0)
