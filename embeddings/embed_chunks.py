from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from embeddings.embedding_model import BgeM3EmbeddingModel, get_embedding_model


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddedChunk:
    """Embedding-ready chunk plus dense vector and retrieval metadata."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _chunk_text(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "text"):
        return str(getattr(chunk, "text") or "")
    if hasattr(chunk, "page_content"):
        return str(getattr(chunk, "page_content") or "")
    raise TypeError(f"Unsupported chunk type: {type(chunk)!r}")


def _chunk_metadata(chunk: object) -> dict[str, Any]:
    if isinstance(chunk, str):
        return {}
    metadata = getattr(chunk, "metadata", None)
    return dict(metadata or {})


def _stable_chunk_id(text: str, metadata: dict[str, Any], index: int) -> str:
    explicit_id = metadata.get("chunk_id") or metadata.get("id")
    if explicit_id:
        return str(explicit_id)
    identity = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", metadata.get("source_page", ""))),
            str(metadata.get("chunk_index", index)),
            text,
        ]
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
    return f"chunk_{index}_{digest}"


def _enrichment_flags(text: str, metadata: dict[str, Any]) -> dict[str, bool]:
    return {
        "contains_chart": bool(
            metadata.get("contains_chart")
            or metadata.get("contains_chart_description")
            or "[CHART DESCRIPTION]" in text
        ),
        "contains_table": bool(metadata.get("contains_table") or "|---" in text or "Table summary:" in text),
        "contains_csv_semantic_sentence": bool(metadata.get("source_type") == "csv" or metadata.get("type") == "csv_row"),
        "contains_diagram": bool(metadata.get("type") == "diagram" or "diagram" in text.lower()),
    }


def _batched(values: Sequence[object], batch_size: int) -> Iterable[tuple[int, Sequence[object]]]:
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]


class ChunkEmbedder:
    """Batch embeds multimodal-enriched chunks with BGE-M3."""

    def __init__(
        self,
        model: BgeM3EmbeddingModel | None = None,
        batch_size: int | None = None,
        max_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self.model = model or get_embedding_model()
        self.batch_size = batch_size or self.model.settings.batch_size
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def embed_chunks(self, chunks: Sequence[object]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        total = len(chunks)
        logger.info("Embedding %s enriched chunks with BGE-M3", total)

        for start, batch in _batched(chunks, self.batch_size):
            texts = [_chunk_text(chunk).strip() for chunk in batch]
            metadatas = [_chunk_metadata(chunk) for chunk in batch]
            vectors = self._embed_with_retry(texts, batch_number=start // self.batch_size + 1)

            for offset, (chunk, text, metadata, vector) in enumerate(zip(batch, texts, metadatas, vectors)):
                if not text:
                    continue
                absolute_index = start + offset
                enriched_metadata = {
                    **metadata,
                    **_enrichment_flags(text, metadata),
                    "chunk_id": _stable_chunk_id(text, metadata, absolute_index),
                    "embedding_model": self.model.settings.model_name_or_path,
                    "embedding_backend": self.model.backend or self.model.settings.backend,
                    "embedding_dimension": len(vector),
                }
                embedded.append(
                    EmbeddedChunk(
                        id=str(enriched_metadata["chunk_id"]),
                        text=text,
                        embedding=vector,
                        metadata=enriched_metadata,
                    )
                )
            logger.info("Embedded chunks %s-%s of %s", start + 1, min(start + len(batch), total), total)

        return embedded

    async def aembed_chunks(self, chunks: Sequence[object]) -> list[EmbeddedChunk]:
        return await asyncio.to_thread(self.embed_chunks, chunks)

    def _embed_with_retry(self, texts: Sequence[str], batch_number: int) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return self.model.embed_documents(texts, batch_size=self.batch_size)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "BGE-M3 embedding batch %s failed on attempt %s/%s: %s",
                    batch_number,
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_sleep_seconds * attempt)
        assert last_error is not None
        raise last_error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_chunks = [
        "Revenue increased steadily from Q1 to Q4.",
        "North America generated the highest sales.",
        "[CHART DESCRIPTION] Sales grew 18 percent in Q4.",
    ]

    embedder = ChunkEmbedder()
    results = embedder.embed_chunks(sample_chunks)

    print("\n===== EMBEDDING TEST SUCCESS =====")

    print(f"\nTotal embedded chunks: {len(results)}")

    print("\nFirst embedded chunk:\n")

    first = results[0]

    print("Chunk ID:")
    print(first.id)

    print("\nChunk Text:")
    print(first.text)

    print("\nEmbedding Dimension:")
    print(len(first.embedding))

    print("\nMetadata:")
    print(first.metadata)
