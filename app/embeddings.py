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

import gc
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)

BGE_MODEL_NAME = "all-MiniLM-L6-v2"
BGE_MODEL_ID = BGE_MODEL_NAME
BGE_QUERY_INSTRUCTION = ""
BGE_EMBEDDING_DIMENSIONS = 384
BGE_CACHE_FOLDER = str(Path(os.getenv("BGE_M3_CACHE_FOLDER", "hf_cache_v2")).resolve())
BGE_MAX_LENGTH = 512

os.environ.setdefault("HF_HOME", BGE_CACHE_FOLDER)
os.environ.setdefault("HF_HUB_CACHE", BGE_CACHE_FOLDER)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Load local SentenceTransformer model
logger = logging.getLogger(__name__)
logger.info("Initializing local SentenceTransformer('all-MiniLM-L6-v2')...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Local SentenceTransformer model loaded successfully.")


def get_query_vector(query_text: str) -> list[float]:
    """Embed a query with local all-MiniLM-L6-v2."""
    vector = model.encode(query_text).tolist()
    return [float(x) for x in vector]


class PureTorchBgeEmbeddings:
    """Small compatibility wrapper for existing retriever calls."""

    def embed_query(self, text: str) -> list[float]:
        return get_query_vector(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [get_query_vector(text) for text in texts]


@lru_cache(maxsize=1)
def get_bge_embeddings() -> PureTorchBgeEmbeddings:
    return PureTorchBgeEmbeddings()

