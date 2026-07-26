from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from vectordb.fastembed_runtime import SafeSparseEncoder
from vectordb.qdrant_client_manager import QdrantSettings, get_qdrant_client as build_managed_qdrant_client

from embeddings.embedding_model import BgeM3EmbeddingModel, EmbeddingModelSettings
from ingest_data import (
    DENSE_VECTOR_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    HF_CACHE_DIR,
    _ensure_local_model,
    _stable_chunk_id,
    parse_sources,
)
from ingestion.entity_metadata import enrich_records_with_cross_references
from ingestion.parent_child import attach_parent_context


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
VISUAL_CAPTION_CACHE_DIR = PROJECT_ROOT / "data_cache" / "visual_captions"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "conversational_rag")
DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")

BGE_MODEL_ID = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_LOCAL_DIR = PROJECT_ROOT / "hf_models_v2" / "bge-m3"
BGE_DEVICE = os.getenv("BGE_M3_DEVICE", "cpu")
BM25_MODEL_NAME = os.getenv("FASTEMBED_BM25_MODEL", "Qdrant/bm25")

UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "32"))

logger = logging.getLogger(__name__)


def qdrant_client() -> QdrantClient:
    settings = QdrantSettings(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
    )
    logger.info("Connecting to Qdrant server at %s:%s", QDRANT_HOST, QDRANT_PORT)
    return build_managed_qdrant_client(settings)


def recreate_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        logger.warning("Deleting old collection: %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    try:
        sparse_params = models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True),
            modifier=models.Modifier.IDF,
        )
    except Exception:
        sparse_params = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))

    logger.info("Creating dual-vector collection: %s", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: sparse_params,
        },
    )

    for field_name, field_schema in {
        "source": models.PayloadSchemaType.KEYWORD,
        "metadata.document_type": models.PayloadSchemaType.KEYWORD,
        "metadata.source_file": models.PayloadSchemaType.KEYWORD,
        "metadata.image_name": models.PayloadSchemaType.KEYWORD,
        "metadata.figure_id": models.PayloadSchemaType.KEYWORD,
        "metadata.entity_id": models.PayloadSchemaType.KEYWORD,
        "metadata.entity_ids": models.PayloadSchemaType.KEYWORD,
        "metadata.cross_reference": models.PayloadSchemaType.KEYWORD,
        "metadata.cross_references": models.PayloadSchemaType.KEYWORD,
        "metadata.parent_id": models.PayloadSchemaType.KEYWORD,
        "metadata.contains_chart": models.PayloadSchemaType.BOOL,
        "metadata.contains_table": models.PayloadSchemaType.BOOL,
        "text": models.PayloadSchemaType.TEXT,
    }.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            logger.debug("Payload index %s skipped: %s", field_name, exc)


def load_bge_embedder() -> BgeM3EmbeddingModel:
    model_path = _ensure_local_model(BGE_MODEL_ID, BGE_LOCAL_DIR)
    logger.info("Loading BGE-M3 dense encoder from %s", model_path)
    return BgeM3EmbeddingModel(
        EmbeddingModelSettings(
            model_name_or_path=model_path,
            device=BGE_DEVICE,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_sequence_length=EMBEDDING_MAX_LENGTH,
            embedding_dimension=DENSE_VECTOR_SIZE,
            normalize_embeddings=True,
            cache_folder=HF_CACHE_DIR,
        )
    )


class Bm25SparseEncoder(SafeSparseEncoder):
    """Backward-compatible alias around the shared safe FastEmbed sparse encoder."""

    def __init__(self, model_name: str = BM25_MODEL_NAME) -> None:
        super().__init__(model_name=model_name)


def _tokenize_for_sparse(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _token_to_sparse_index(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def _local_sparse_vector(text: str) -> models.SparseVector:
    counts: dict[int, float] = {}
    for token in _tokenize_for_sparse(text):
        index = _token_to_sparse_index(token)
        counts[index] = counts.get(index, 0.0) + 1.0
    if not counts:
        return models.SparseVector(indices=[], values=[])
    return models.SparseVector(
        indices=list(counts.keys()),
        values=[1.0 + value**0.5 for value in counts.values()],
    )


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def load_text_csv_pdf_records(source_paths: Iterable[Path]) -> list[dict[str, Any]]:
    logger.info("Parsing text, PDF, and CSV records without vision API calls")
    raw_records = parse_sources(source_paths, enrich_pdf_visuals=False)
    records: list[dict[str, Any]] = []

    for record in raw_records:
        text = clean_text(record.get("text"))
        if not text:
            continue
        source = clean_text(record.get("source")) or "unknown"
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("source", source)
        metadata.setdefault("source_file", source)
        metadata.setdefault("document_type", metadata.get("document_type", "text"))
        metadata.setdefault("contains_table", metadata.get("document_type") == "csv")
        metadata.setdefault("contains_chart", False)
        records.append({"text": text, "source": source, "metadata": metadata})

    logger.info("Loaded %s text/PDF/CSV records", len(records))
    return records


def visual_source_file(image_name: str, image_path: str) -> str:
    for value in (image_name, image_path):
        stem = Path(value).stem
        marker = "_figure_"
        if marker in stem:
            return f"{stem.split(marker, 1)[0]}.pdf"
    return "visual_caption_cache"


def load_visual_caption_records(cache_dir: Path = VISUAL_CAPTION_CACHE_DIR) -> list[dict[str, Any]]:
    if not cache_dir.exists():
        logger.warning("Visual caption cache directory missing: %s", cache_dir)
        return []

    records: list[dict[str, Any]] = []
    cache_files = sorted(cache_dir.glob("*.json"))
    logger.info("Loading %s visual caption cache files", len(cache_files))

    for index, cache_file in enumerate(cache_files, start=1):
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping unreadable cache file %s: %s", cache_file, exc)
            continue

        structured_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        structured_metadata = (
            structured_payload.get("metadata")
            if isinstance(structured_payload.get("metadata"), dict)
            else {}
        )

        caption = clean_text(structured_payload.get("text")) or clean_text(payload.get("caption"))
        if not caption:
            continue

        image_name = (
            clean_text(structured_metadata.get("file_name"))
            or clean_text(payload.get("image_name"))
            or Path(clean_text(payload.get("image_path"))).name
        )
        image_path = clean_text(payload.get("image_path"))
        image_hash = clean_text(payload.get("image_hash")) or cache_file.stem
        figure_id = clean_text(structured_metadata.get("figure_id")) or clean_text(payload.get("figure_id")) or "Unknown"
        source_file = visual_source_file(image_name, image_path)
        text = (
            "[VISUAL FIGURE DESCRIPTION]\n"
            f"Source file: {source_file}\n"
            f"Image name: {image_name}\n"
            f"Image hash: {image_hash}\n"
            f"Figure ID: {figure_id}\n\n"
            f"{caption}\n"
            "[/VISUAL FIGURE DESCRIPTION]"
        )
        metadata = {
            "chunk_id": _stable_chunk_id(text),
            "document_type": "pdf_visual",
            "type": "visual_caption",
            "source": source_file,
            "source_file": source_file,
            "image_name": image_name,
            "image_path": image_path,
            "image_hash": image_hash,
            "figure_id": figure_id,
            "caption_cache_path": str(cache_file),
            "vision_model": clean_text(payload.get("model")),
            "visual_caption_index": index,
            "contains_chart": True,
            "contains_diagram": True,
            "contains_table": False,
        }
        records.append({"text": text, "source": source_file, "metadata": metadata})

    logger.info("Loaded %s visual caption records", len(records))
    return records


def load_all_records(source_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = attach_parent_context(enrich_records_with_cross_references([
        *load_text_csv_pdf_records(source_paths),
        *load_visual_caption_records(),
    ]))
    logger.info("Total records ready for dense+sparse indexing: %s", len(records))
    return records


def record_batches(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def build_points(
    records: list[dict[str, Any]],
    dense_vectors: list[list[float]],
    sparse_vectors: list[models.SparseVector],
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for record, dense, sparse in zip(records, dense_vectors, sparse_vectors):
        text = clean_text(record["text"])
        metadata = dict(record.get("metadata") or {})
        chunk_id = clean_text(metadata.get("chunk_id")) or _stable_chunk_id(text) or str(uuid.uuid4())
        source = clean_text(record.get("source")) or clean_text(metadata.get("source")) or "unknown"

        if len(dense) != DENSE_VECTOR_SIZE:
            raise ValueError(f"Expected dense dim {DENSE_VECTOR_SIZE}, got {len(dense)} for {source}")

        payload = {
            "text": text,
            "page_content": text,
            "source": source,
            "metadata": metadata,
        }
        if not clean_text(payload["text"]):
            raise ValueError(f"Cannot upsert record without root payload['text']; chunk_id={chunk_id}")
        points.append(
            models.PointStruct(
                id=chunk_id,
                vector={
                    DENSE_VECTOR_NAME: [float(value) for value in dense],
                    SPARSE_VECTOR_NAME: sparse,
                },
                payload=payload,
            )
        )
    return points


def upsert_batch(client: QdrantClient, points: list[models.PointStruct], uploaded: int, total: int) -> int:
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    uploaded += len(points)
    logger.info("Upserted %s/%s dense+sparse chunks", uploaded, total)
    return uploaded


def reindex(source_paths: Iterable[Path]) -> int:
    records = load_all_records(source_paths)
    if not records:
        logger.warning("No records found. Nothing to index.")
        return 0

    client = qdrant_client()
    try:
        recreate_collection(client)
        dense_encoder = load_bge_embedder()
        sparse_encoder = Bm25SparseEncoder()

        uploaded = 0
        total = len(records)
        for batch_index, batch in enumerate(record_batches(records, UPSERT_BATCH_SIZE), start=1):
            texts = [record["text"] for record in batch]
            logger.info("Encoding batch %s containing %s chunks", batch_index, len(batch))
            dense_vectors = dense_encoder.embed_documents(texts, batch_size=EMBEDDING_BATCH_SIZE)
            sparse_vectors = sparse_encoder.encode_documents(texts)
            points = build_points(batch, dense_vectors, sparse_vectors)
            uploaded = upsert_batch(client, points, uploaded, total)

        exact_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        logger.info("Sparse reindex complete. Uploaded=%s Qdrant exact_count=%s", uploaded, exact_count)
        return uploaded
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant with BGE-M3 dense vectors plus BM25 sparse vectors.")
    parser.add_argument("sources", nargs="*", default=[str(DATA_DIR)], help="Files/directories containing text, PDF, and CSV data.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    source_paths = [Path(source).expanduser().resolve() for source in args.sources]
    reindex(source_paths)


if __name__ == "__main__":
    main()
