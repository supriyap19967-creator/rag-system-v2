from __future__ import annotations

import sys
import types
import datasets

# Mock sentence_transformers trainer, training_args, cross_encoder, and sparse_encoder to bypass Trainer imports
sys.modules['sentence_transformers.trainer'] = types.ModuleType('sentence_transformers.trainer')
sys.modules['sentence_transformers.trainer'].SentenceTransformerTrainer = None

sys.modules['sentence_transformers.training_args'] = types.ModuleType('sentence_transformers.training_args')
sys.modules['sentence_transformers.training_args'].SentenceTransformerTrainingArguments = None
sys.modules['sentence_transformers.training_args'].BatchSamplers = None
sys.modules['sentence_transformers.training_args'].MultiDatasetBatchSamplers = None

sys.modules['sentence_transformers.sparse_encoder'] = types.ModuleType('sentence_transformers.sparse_encoder')
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoder = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderModelCardData = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainer = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainingArguments = None

sys.modules['sentence_transformers.cross_encoder'] = types.ModuleType('sentence_transformers.cross_encoder')
sys.modules['sentence_transformers.cross_encoder'].CrossEncoder = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderModelCardData = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainer = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainingArguments = None

from sentence_transformers import SentenceTransformer

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize local SentenceTransformer model
logger.info("Initializing local SentenceTransformer('all-MiniLM-L6-v2') inside embeddings/embedding_model.py...")
_local_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Local SentenceTransformer model loaded successfully.")


@dataclass(frozen=True, slots=True)
class EmbeddingModelSettings:
    """Environment-backed embedding configuration."""
    model_name_or_path: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 128
    max_sequence_length: int = 512
    embedding_dimension: int = 384
    normalize_embeddings: bool = True
    cache_folder: Path = Path("hf_cache_v2").resolve()
    backend: str = "local"


class BgeM3EmbeddingModel:
    """SentenceTransformer-backed embedding model that behaves like BgeM3EmbeddingModel for compatibility."""

    def __init__(self, settings: EmbeddingModelSettings | None = None) -> None:
        self.settings = settings or EmbeddingModelSettings()

    @property
    def dimension(self) -> int:
        return 384

    @property
    def backend(self) -> str:
        return "local"

    def embed_documents(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed enriched chunks using local model."""
        clean_texts = [str(text or "").strip() for text in texts]
        if not clean_texts:
            return []
        
        # Encode texts
        embeddings = _local_model.encode(clean_texts)
        return [[float(x) for x in emb.tolist()] for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a conversational retrieval query into the same space."""
        vector = _local_model.encode(query).tolist()
        return [float(x) for x in vector]

    async def aembed_documents(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts, batch_size)

    async def aembed_query(self, query: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, query)


@lru_cache(maxsize=1)
def get_embedding_model() -> BgeM3EmbeddingModel:
    return BgeM3EmbeddingModel()


def _embedding_shape(vectors: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not vectors:
        return (0, 0)
    return (len(vectors), len(vectors[0]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("Testing BGE-M3 model loading...")

    sample_texts = [
        "Revenue increased steadily from Q1 to Q4.",
        "The company showed strong financial growth.",
        "Cats are sleeping on the sofa.",
    ]

    model = get_embedding_model()
    embeddings = model.embed_documents(sample_texts)

    print("Embedding generation successful!")

    print("\nEmbedding shape:")
    print(_embedding_shape(embeddings))

    print("\nFirst vector sample:")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()
