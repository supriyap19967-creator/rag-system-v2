from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Sequence

from embeddings.embed_chunks import ChunkEmbedder, EmbeddedChunk
from ingestion.schemas import Chunk
from app.multimodal_assets import ASSET_FIELDS
from vectordb.metadata_schema import normalize_payload
from vectordb.qdrant_client_manager import (
    QdrantSettings,
    ensure_hybrid_collection,
    get_qdrant_async_client,
    get_qdrant_client,
)


logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "conversational_rag"


def stable_point_id(chunk: EmbeddedChunk) -> str:
    """Generate deterministic UUIDv5 IDs from raw chunk text content."""

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.text))


def build_point(chunk: EmbeddedChunk):
    from qdrant_client import models

    payload = normalize_payload(chunk.text, {**chunk.metadata, "chunk_id": chunk.id}).to_qdrant_payload()
    payload["chunk_id"] = chunk.id
    payload["source"] = str(chunk.metadata.get("source") or payload.get("source_file") or "")
    payload["contains_chart"] = bool(chunk.metadata.get("contains_chart"))
    payload["contains_table"] = bool(chunk.metadata.get("contains_table"))
    payload["contains_figure"] = bool(payload.get("contains_figure") or chunk.metadata.get("contains_figure"))
    payload["contains_image"] = bool(payload.get("contains_image") or chunk.metadata.get("contains_image"))
    payload["contains_csv"] = bool(payload.get("contains_csv") or chunk.metadata.get("contains_csv"))
    payload["embedding_model"] = str(chunk.metadata.get("embedding_model") or "")
    payload["page_content"] = chunk.text
    payload["text"] = chunk.text
    if not str(payload["text"] or "").strip():
        raise ValueError(f"Cannot upsert chunk without root payload['text']; chunk_id={chunk.id}")
    payload.setdefault("metadata", {})
    payload["metadata"]["chunk_id"] = chunk.id
    payload["metadata"]["source"] = payload["source"]
    payload["metadata"]["contains_chart"] = payload["contains_chart"]
    payload["metadata"]["contains_table"] = payload["contains_table"]
    payload["metadata"]["contains_figure"] = payload["contains_figure"]
    payload["metadata"]["contains_image"] = payload["contains_image"]
    payload["metadata"]["contains_csv"] = payload["contains_csv"]
    payload["metadata"]["embedding_model"] = payload["embedding_model"]
    for key in ASSET_FIELDS:
        value = payload.get(key, chunk.metadata.get(key))
        if value not in ("", None, [], {}):
            payload[key] = value
            payload["metadata"][key] = value
    for key in ("chapter_number", "chapter_title", "section_title", "h1", "h2", "h3"):
        if payload.get(key):
            payload["metadata"].setdefault(key, payload[key])
    return models.PointStruct(
        id=stable_point_id(chunk),
        vector={"dense": chunk.embedding},
        payload=payload,
    )


class QdrantVectorIngester:
    """Batch upload BGE-M3 embedded chunks into Qdrant."""

    def __init__(
        self,
        collection_name: str | None = None,
        batch_size: int = 64,
        max_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
        client: object | None = None,
        settings: QdrantSettings | None = None,
    ) -> None:
        self.settings = settings or QdrantSettings()
        self.collection_name = collection_name or self.settings.collection_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.client = client or get_qdrant_client(self.settings)
        ensure_hybrid_collection(self.client, self.collection_name)

    def ingest(self, chunks: Sequence[EmbeddedChunk]) -> int:
        total = len(chunks)
        uploaded = 0
        logger.info("Uploading %s embedded chunks to Qdrant collection %s", total, self.collection_name)

        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            points = [build_point(chunk) for chunk in batch]
            self._upsert_with_retry(points, batch_number=start // self.batch_size + 1)
            uploaded += len(points)
            logger.info("Uploaded Qdrant points %s-%s of %s", start + 1, uploaded, total)
        return uploaded

    def _upsert_with_retry(self, points: Sequence[object], batch_number: int) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                self.client.upsert(collection_name=self.collection_name, points=list(points), wait=True)
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Qdrant upsert batch %s failed on attempt %s/%s: %s",
                    batch_number,
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_sleep_seconds * attempt)
        assert last_error is not None
        raise last_error


class AsyncQdrantVectorIngester:
    """Async Qdrant uploader for high-throughput ingestion jobs."""

    def __init__(
        self,
        collection_name: str | None = None,
        batch_size: int = 64,
        client: object | None = None,
        settings: QdrantSettings | None = None,
    ) -> None:
        self.settings = settings or QdrantSettings()
        self.collection_name = collection_name or self.settings.collection_name
        self.batch_size = batch_size
        self.client = client or get_qdrant_async_client(self.settings)

    async def ingest(self, chunks: Sequence[EmbeddedChunk]) -> int:
        uploaded = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            points = [build_point(chunk) for chunk in batch]
            await self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            uploaded += len(points)
            logger.info("Uploaded async Qdrant points %s-%s of %s", start + 1, uploaded, len(chunks))
            await asyncio.sleep(0)
        return uploaded


def _sample_chunks() -> list[object]:
    return [
        Chunk(
            text="Revenue increased steadily from Q1 to Q4.",
            metadata={
                "source": "PDF_Text",
                "chunk_id": "sample_chunk_1",
                "contains_chart": False,
                "contains_table": False,
            },
        ),
        Chunk(
            text="North America generated the highest sales.",
            metadata={
                "source": "PDF_Table",
                "chunk_id": "sample_chunk_2",
                "contains_chart": False,
                "contains_table": True,
            },
        ),
        Chunk(
            text="[CHART DESCRIPTION] Sales grew 18 percent in Q4.",
            metadata={
                "source": "Qwen_VL_Chart",
                "chunk_id": "sample_chunk_3",
                "contains_chart": True,
                "contains_table": False,
            },
        ),
    ]


def ingest_vectors(chunks: Sequence[object] | None = None, collection_name: str = DEFAULT_COLLECTION_NAME) -> int:
    """Embed chunks with real BGE-M3 vectors and upsert them into Qdrant."""

    if chunks is None:
        raise ValueError(
            "No chunks supplied for vector ingestion. "
            "Use ingest_data.py for document ingestion, or pass real parsed chunks explicitly."
        )

    source_chunks = list(chunks)
    embedder = ChunkEmbedder()
    embedded_chunks = embedder.embed_chunks(source_chunks)

    bad_dimensions = [
        len(chunk.embedding)
        for chunk in embedded_chunks
        if len(chunk.embedding) != embedder.model.settings.embedding_dimension
    ]
    if bad_dimensions:
        raise ValueError(
            "Embedding dimension mismatch before Qdrant upsert: "
            f"expected {embedder.model.settings.embedding_dimension}, got {bad_dimensions}"
        )

    logger.info(
        "Generated %s real BGE-M3 embeddings with dimension %s",
        len(embedded_chunks),
        embedder.model.settings.embedding_dimension,
    )
    return QdrantVectorIngester(collection_name=collection_name).ingest(embedded_chunks)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    raise SystemExit(
        "vectordb.ingest_vectors no longer seeds sample data by default. "
        "Run `python ingest_data.py` to ingest real documents into Qdrant."
    )
