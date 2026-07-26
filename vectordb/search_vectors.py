from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from vectordb.qdrant_client_manager import QdrantSettings, get_qdrant_client


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]
    payload: dict[str, Any]


def build_metadata_filter(filters: dict[str, Any] | None):
    """Build a Qdrant filter from simple equality/range metadata constraints."""

    if not filters:
        return None

    from qdrant_client import models

    conditions = []
    for field, value in filters.items():
        if value is None:
            continue
        if isinstance(value, dict):
            conditions.append(
                models.FieldCondition(
                    key=field,
                    range=models.Range(
                        gte=value.get("gte"),
                        gt=value.get("gt"),
                        lte=value.get("lte"),
                        lt=value.get("lt"),
                    ),
                )
            )
        elif isinstance(value, (list, tuple, set)):
            conditions.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchAny(any=list(value)),
                )
            )
        else:
            conditions.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value),
                )
            )
    return models.Filter(must=conditions) if conditions else None


class QdrantSearcher:
    """Semantic similarity search over multimodal-enriched Qdrant chunks."""

    def __init__(
        self,
        collection_name: str | None = None,
        client: object | None = None,
        settings: QdrantSettings | None = None,
    ) -> None:
        self.settings = settings or QdrantSettings()
        self.collection_name = collection_name or self.settings.collection_name
        self.client = client or get_qdrant_client(self.settings)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        qdrant_filter = build_metadata_filter(filters)
        query_kwargs = {
            "collection_name": self.collection_name,
            "query": list(query_vector),
            "query_filter": qdrant_filter,
            "limit": top_k,
            "score_threshold": score_threshold,
            "with_payload": True,
            "with_vectors": False,
        }
        if self._has_named_vector("dense"):
            query_kwargs["using"] = "dense"
        response = self.client.query_points(**query_kwargs)
        points = response.points or []
        return [self._to_result(point) for point in points]

    def _has_named_vector(self, vector_name: str) -> bool:
        try:
            collection = self.client.get_collection(self.collection_name)
            vectors = collection.config.params.vectors
            return isinstance(vectors, dict) and vector_name in vectors
        except Exception as exc:
            logger.debug("Could not inspect Qdrant vector config for %s: %s", self.collection_name, exc)
            return False

    @staticmethod
    def _payload_text(payload: dict[str, Any]) -> str:
        nested_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        return str(
            payload.get("text")
            or payload.get("page_content")
            or payload.get("content")
            or nested_payload.get("text")
            or ""
        ).strip()

    @staticmethod
    def _to_result(point: object) -> SearchResult:
        payload = dict(getattr(point, "payload", None) or {})
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "chunk_id": payload.get("chunk_id"),
                "source_file": payload.get("source_file"),
                "document_type": payload.get("document_type"),
                "page": payload.get("page"),
                "contains_chart": payload.get("contains_chart", False),
                "contains_table": payload.get("contains_table", False),
                "contains_diagram": payload.get("contains_diagram", False),
                "image_reference": payload.get("image_reference", ""),
            }
        )
        return SearchResult(
            id=str(getattr(point, "id", "")),
            text=QdrantSearcher._payload_text(payload),
            score=float(getattr(point, "score", 0.0)),
            metadata=metadata,
            payload=payload,
        )


if __name__ == "__main__":
    import logging

    from embeddings.embedding_model import get_embedding_model

    logging.basicConfig(level=logging.INFO)

    print("Starting vector search...")

    query = "Which chunk discusses Q4 revenue growth?"
    query_vector = get_embedding_model().embed_query(query)
    results = QdrantSearcher(collection_name="conversational_rag").search(query_vector=query_vector, top_k=5)

    print(f"\nTotal results: {len(results)}")
    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Score: {result.score}")
        print(f"Source: {result.metadata.get('source') or result.payload.get('source')}")
        print(result.text[:500])

    print("\nVector search completed!")
