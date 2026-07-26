from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embeddings.embed_chunks import EmbeddedChunk
from embeddings.embedding_model import BgeM3EmbeddingModel, get_embedding_model


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class FaissVectorStore:
    """Local FAISS vector store for dense BGE-M3 retrieval.

    The sidecar JSON keeps source text and metadata available for conversational
    RAG while FAISS handles fast cosine-similarity search over normalized vectors.
    """

    def __init__(self, persist_dir: str | Path = "Data/vectorstores/bge_m3_faiss") -> None:
        self.persist_dir = Path(persist_dir)
        self.index_path = self.persist_dir / "index.faiss"
        self.documents_path = self.persist_dir / "documents.json"
        self._index: object | None = None
        self._documents: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self._documents)

    def build(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store with zero chunks.")

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required for FaissVectorStore.") from exc

        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self._index = index
        self._documents = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": self._json_safe_metadata(chunk.metadata),
            }
            for chunk in chunks
        ]
        logger.info("Built FAISS BGE-M3 vector store with %s chunks", len(self._documents))

    def add(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if self._index is None:
            self.build(chunks)
            return
        if not chunks:
            return

        import faiss

        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype="float32")
        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._documents.extend(
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": self._json_safe_metadata(chunk.metadata),
            }
            for chunk in chunks
        )
        logger.info("Added %s chunks to FAISS BGE-M3 vector store", len(chunks))

    def save(self) -> None:
        if self._index is None:
            raise ValueError("No FAISS index is loaded or built.")

        import faiss

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.documents_path.write_text(
            json.dumps({"documents": self._documents}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved FAISS BGE-M3 vector store to %s", self.persist_dir)

    def load(self) -> None:
        if not self.index_path.exists() or not self.documents_path.exists():
            raise FileNotFoundError(f"Missing FAISS store files in {self.persist_dir}")

        import faiss

        self._index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.documents_path.read_text(encoding="utf-8"))
        self._documents = list(payload.get("documents") or [])
        logger.info("Loaded FAISS BGE-M3 vector store with %s chunks", len(self._documents))

    def search_by_vector(self, embedding: Sequence[float], top_k: int = 5) -> list[RetrievalResult]:
        if self._index is None:
            self.load()
        if self._index is None:
            raise ValueError("No FAISS index is loaded.")

        import faiss

        query = np.asarray([embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, top_k)

        results: list[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self._documents):
                continue
            document = self._documents[int(index)]
            results.append(
                RetrievalResult(
                    id=str(document["id"]),
                    text=str(document["text"]),
                    score=float(score),
                    metadata=dict(document.get("metadata") or {}),
                )
            )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        model: BgeM3EmbeddingModel | None = None,
    ) -> list[RetrievalResult]:
        embedder = model or get_embedding_model()
        query_vector = embedder.embed_query(query)
        return self.search_by_vector(query_vector, top_k=top_k)

    @staticmethod
    def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            elif isinstance(value, (list, tuple)):
                safe[key] = [str(item) for item in value]
            elif isinstance(value, dict):
                safe[key] = {str(k): str(v) for k, v in value.items()}
            else:
                safe[key] = str(value)
        return safe

    def to_dict(self) -> dict[str, Any]:
        return {"persist_dir": str(self.persist_dir), "count": self.count}
