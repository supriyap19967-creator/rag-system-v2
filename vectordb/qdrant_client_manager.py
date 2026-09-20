from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    """Environment-backed Qdrant connection settings."""

    collection_name: str = os.getenv("QDRANT_COLLECTION", "conversational_rag")
    path: str = (os.getenv("QDRANT_PATH") or "./qdrant_db").strip()
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    url: str = os.getenv("QDRANT_URL", "").strip()
    api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    prefer_grpc: bool = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {"1", "true", "yes"}
    timeout_seconds: float = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "3.0"))
    pool_size: int = int(os.getenv("QDRANT_GRPC_POOL_SIZE", "4"))
    max_connections: int = int(os.getenv("QDRANT_HTTP_MAX_CONNECTIONS", "24"))
    max_keepalive_connections: int = int(os.getenv("QDRANT_HTTP_MAX_KEEPALIVE", "12"))


def _httpx_limits(settings: QdrantSettings) -> object | None:
    try:
        import httpx
    except ImportError:
        return None
    return httpx.Limits(
        max_connections=max(int(settings.max_connections), 1),
        max_keepalive_connections=max(int(settings.max_keepalive_connections), 1),
    )


def _client_kwargs(settings: QdrantSettings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "api_key": settings.api_key or None,
        "prefer_grpc": settings.prefer_grpc,
        "timeout": settings.timeout_seconds,
        "pool_size": max(int(settings.pool_size), 1),
        "check_compatibility": False,
    }
    limits = _httpx_limits(settings)
    if limits is not None:
        kwargs["limits"] = limits
        kwargs.pop("pool_size", None)
    if settings.path:
        kwargs["path"] = str(Path(settings.path).expanduser())
        kwargs.pop("prefer_grpc", None)
        kwargs.pop("pool_size", None)
        kwargs.pop("host", None)
        kwargs.pop("port", None)
        kwargs.pop("grpc_port", None)
    elif settings.url:
        kwargs["url"] = settings.url
    else:
        kwargs["host"] = settings.host
        kwargs["port"] = settings.port
        kwargs["grpc_port"] = settings.grpc_port
    kwargs["timeout"] = 3.0
    return kwargs


def _create_qdrant_client(settings: QdrantSettings):
    from qdrant_client import QdrantClient

    kwargs = _client_kwargs(settings)
    for attempt in range(3):
        try:
            return QdrantClient(**kwargs)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "already accessed" in err_msg or "already locked" in err_msg or "permission denied" in err_msg:
                if attempt < 2:
                    time.sleep(0.5)
                    continue
                # Use persistent secondary storage folder instead of failing server connection
                fallback_path = os.path.abspath("./qdrant_db_local")
                clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("path", "location", "url", "host", "port", "grpc_port")}
                clean_kwargs["path"] = fallback_path
                try:
                    return QdrantClient(**clean_kwargs)
                except Exception:
                    pass
                clean_kwargs.pop("path", None)
                clean_kwargs["location"] = ":memory:"
                return QdrantClient(**clean_kwargs)

            unsupported = {key for key in ("limits", "pool_size") if key in kwargs}
            if unsupported and attempt == 0:
                for key in unsupported:
                    kwargs.pop(key, None)
                continue
            raise exc


def ensure_hybrid_collection(client: object, collection_name: str = "conversational_rag") -> None:
    """Ensure the BGE-M3 hybrid collection exists before upsert/search."""

    from qdrant_client.http import models as qmodels

    from vectordb.create_collection import create_payload_indexes

    if client.collection_exists(collection_name):
        create_payload_indexes(client=client, collection_name=collection_name)
        return

    print(f"Collection '{collection_name}' not found. Creating brand new hybrid layout...")
    try:
        sparse_params = qmodels.SparseVectorParams(
            index=qmodels.SparseIndexParams(on_disk=True),
            modifier=qmodels.Modifier.IDF,
        )
    except Exception:
        sparse_params = qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=True))

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=384,
                distance=qmodels.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": sparse_params,
        },
    )
    print(f"Collection '{collection_name}' created successfully.")
    create_payload_indexes(client=client, collection_name=collection_name)


@lru_cache(maxsize=1)
def get_qdrant_client(settings: QdrantSettings | None = None):
    """Return a singleton-style synchronous Qdrant client."""

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("Install qdrant-client to use Qdrant vector storage.") from exc

    settings = settings or QdrantSettings(path=os.path.abspath("./qdrant_db"))
    logger.info("Initializing Qdrant client for collection %s", settings.collection_name)
    client = _create_qdrant_client(settings)
    ensure_hybrid_collection(client, settings.collection_name)
    return client


@lru_cache(maxsize=1)
def get_qdrant_async_client(settings: QdrantSettings | None = None):
    """Return a singleton-style asynchronous Qdrant client."""

    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as exc:
        raise RuntimeError("Install qdrant-client to use async Qdrant vector storage.") from exc

    settings = settings or QdrantSettings()
    logger.info("Initializing async Qdrant client for collection %s", settings.collection_name)
    kwargs = _client_kwargs(settings)
    try:
        return AsyncQdrantClient(**kwargs)
    except TypeError as exc:
        unsupported = {key for key in ("limits", "pool_size") if key in kwargs}
        if not unsupported:
            raise
        logger.warning(
            "AsyncQdrantClient rejected optional connection settings %s (%s); retrying with core kwargs only",
            sorted(unsupported),
            exc,
        )
        for key in unsupported:
            kwargs.pop(key, None)
        return AsyncQdrantClient(**kwargs)
