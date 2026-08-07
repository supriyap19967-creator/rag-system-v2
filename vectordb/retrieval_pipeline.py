from __future__ import annotations

import os
import sys

# Force root directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from dataclasses import dataclass
from typing import Any

from embeddings.embedding_model import BgeM3EmbeddingModel, get_embedding_model
from gateway_guardrails import GatewayInfrastructure
from vectordb.search_vectors import QdrantSearcher, SearchResult


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalContext:
    query: str
    rewritten_query: str
    results: list[SearchResult]

    def to_llm_context(self) -> list[dict[str, Any]]:
        """Prepare retrieved chunks for answer generation with source attribution."""

        return [
            {
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in self.results
        ]


class ConversationalRetrievalPipeline:
    """Embed conversational queries with BGE-M3 and retrieve enriched chunks from Qdrant."""

    def __init__(
        self,
        embedder: BgeM3EmbeddingModel | None = None,
        searcher: QdrantSearcher | None = None,
        max_history_turns: int = 4,
    ) -> None:
        self.embedder = embedder or get_embedding_model()
        self.searcher = searcher or QdrantSearcher()
        self.max_history_turns = max_history_turns
        self.gateway = GatewayInfrastructure(request_cap=1_000_000)

    def retrieve(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalContext:
        rewritten_query = self._query_with_context(query, conversation_history or [])
        query_vector = self.embedder.embed_query(rewritten_query)
        results = self.searcher.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
        logger.info("Retrieved %s Qdrant chunks for conversational query", len(results))
        return RetrievalContext(query=query, rewritten_query=rewritten_query, results=results)

    def _query_with_context(self, query: str, history: list[dict[str, str]]) -> str:
        if not history:
            return self.gateway._redact_pii(query)[0]
        recent = history[-self.max_history_turns :]
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in recent
            if turn.get("content")
        )
        sanitized_text, _ = self.gateway._redact_pii(
            "Conversation context:\n"
            f"{history_text}\n\n"
            "Current retrieval question:\n"
            f"{query}"
        )
        return sanitized_text.strip()
