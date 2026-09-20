from __future__ import annotations

from typing import Any, List, Optional, TypedDict


class RAGState(TypedDict, total=False):
    """Shared state dictionary passed across all nodes in the LangGraph workflow."""

    question: str
    session_id: str
    structural_intent: str
    locked_entities: List[str]
    condensed_query: str
    retrieved_chunks: List[dict[str, Any]]
    active_image_path: Optional[str]
    active_csv_path: Optional[str]
    generated_answer: str
    confidence: float
    retry_count: int
    is_grounded: bool
    trace_id: Optional[str]
    model_used: str
    error_message: Optional[str]
