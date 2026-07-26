from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_BM25_MODEL_NAME = os.getenv("FASTEMBED_BM25_MODEL", "Qdrant/bm25")
DEFAULT_FASTEMBED_CACHE_DIR = Path(
    os.getenv("FASTEMBED_CACHE_PATH", os.getenv("FASTEMBED_CACHE_DIR", ".fastembed_cache"))
).expanduser()


@dataclass(frozen=True, slots=True)
class FastEmbedRuntimeSettings:
    """Environment-backed FastEmbed initialization settings."""

    model_name: str = DEFAULT_BM25_MODEL_NAME
    cache_dir: Path = DEFAULT_FASTEMBED_CACHE_DIR
    specific_model_path: str = os.getenv("FASTEMBED_MODEL_PATH", "").strip()
    local_files_only: bool = os.getenv("FASTEMBED_LOCAL_FILES_ONLY", "true").lower() not in {"0", "false", "no"}
    allow_network_download: bool = os.getenv("FASTEMBED_ALLOW_NETWORK_DOWNLOAD", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _hf_model_cache_root(cache_dir: Path, model_name: str) -> Path:
    return cache_dir / f"models--{model_name.replace('/', '--')}"


def verify_model_directory(path: Path) -> bool:
    """Return True when a directory contains loadable FastEmbed ONNX assets."""

    directory = path.expanduser().resolve()
    if not directory.is_dir():
        return False
    files = {item.name for item in directory.iterdir() if item.is_file()}
    if "model.onnx" in files or "model_optimized.onnx" in files:
        return True
    if "config.json" in files and any(name.endswith(".onnx") for name in files):
        return True
    return False


def resolve_verified_model_path(settings: FastEmbedRuntimeSettings | None = None) -> Path | None:
    """Resolve a pre-verified on-disk model directory without triggering downloads."""

    settings = settings or FastEmbedRuntimeSettings()
    cache_dir = settings.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if settings.specific_model_path:
        candidate = Path(settings.specific_model_path).expanduser().resolve()
        if verify_model_directory(candidate):
            logger.info("Using verified FastEmbed model directory from FASTEMBED_MODEL_PATH: %s", candidate)
            return candidate
        logger.warning(
            "FASTEMBED_MODEL_PATH does not contain a verified FastEmbed model directory: %s",
            candidate,
        )

    snapshots_root = _hf_model_cache_root(cache_dir, settings.model_name) / "snapshots"
    if snapshots_root.is_dir():
        for snapshot in sorted(snapshots_root.iterdir(), reverse=True):
            if snapshot.is_dir() and verify_model_directory(snapshot):
                logger.info("Using verified FastEmbed snapshot cache: %s", snapshot.resolve())
                return snapshot.resolve()

    return None


def _sparse_embedding_kwargs(settings: FastEmbedRuntimeSettings, verified_path: Path | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_name": settings.model_name,
        "cache_dir": str(settings.cache_dir.expanduser().resolve()),
    }
    if verified_path is not None:
        kwargs["specific_model_path"] = str(verified_path)
        kwargs["local_files_only"] = True
        return kwargs
    if settings.local_files_only and not settings.allow_network_download:
        return {}
    kwargs["local_files_only"] = settings.local_files_only and not settings.allow_network_download
    return kwargs


@lru_cache(maxsize=4)
def build_sparse_text_embedding(model_name: str = DEFAULT_BM25_MODEL_NAME) -> Any | None:
    """Load SparseTextEmbedding from a verified local directory, or return None for fallback."""

    settings = FastEmbedRuntimeSettings(model_name=model_name)
    verified_path = resolve_verified_model_path(settings)
    init_kwargs = _sparse_embedding_kwargs(settings, verified_path)
    if not init_kwargs:
        logger.warning(
            "No verified local FastEmbed model for %s under %s; skipping network download and using sparse fallback",
            settings.model_name,
            settings.cache_dir,
        )
        return None

    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        logger.warning("fastembed is not installed; using deterministic sparse keyword fallback")
        return None

    try:
        model = SparseTextEmbedding(**init_kwargs)
        logger.info(
            "Loaded FastEmbed sparse model %s (local_files_only=%s, specific_model_path=%s)",
            settings.model_name,
            init_kwargs.get("local_files_only"),
            init_kwargs.get("specific_model_path"),
        )
        return model
    except Exception as exc:
        logger.warning(
            "FastEmbed SparseTextEmbedding initialization failed for %s: %s. Using sparse keyword fallback.",
            settings.model_name,
            exc,
        )
        return None


def tokenize_for_sparse(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def token_to_sparse_index(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def local_sparse_vector(text: str) -> Any:
    from qdrant_client import models

    counts: dict[int, float] = {}
    for token in tokenize_for_sparse(text):
        index = token_to_sparse_index(token)
        counts[index] = counts.get(index, 0.0) + 1.0
    if not counts:
        return models.SparseVector(indices=[], values=[])
    return models.SparseVector(
        indices=list(counts.keys()),
        values=[1.0 + value**0.5 for value in counts.values()],
    )


class SafeSparseEncoder:
    """Sparse query/document encoder with verified-local FastEmbed init and deterministic fallback."""

    def __init__(self, model_name: str = DEFAULT_BM25_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = build_sparse_text_embedding(model_name)
        self._fallback = self._model is None

    @property
    def using_fallback(self) -> bool:
        return self._fallback

    def encode_query(self, text: str) -> Any:
        if self._fallback:
            return local_sparse_vector(text)
        sparse_embedding = next(iter(self._model.query_embed(text)))
        from qdrant_client import models

        return models.SparseVector(
            indices=[int(index) for index in sparse_embedding.indices],
            values=[float(value) for value in sparse_embedding.values],
        )

    def encode_documents(self, texts: Sequence[str]) -> list[Any]:
        if self._fallback:
            return [local_sparse_vector(text) for text in texts]
        from qdrant_client import models

        vectors: list[models.SparseVector] = []
        for sparse_embedding in self._model.passage_embed(list(texts)):
            vectors.append(
                models.SparseVector(
                    indices=[int(index) for index in sparse_embedding.indices],
                    values=[float(value) for value in sparse_embedding.values],
                )
            )
        return vectors
