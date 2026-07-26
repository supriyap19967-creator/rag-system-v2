from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from app.multimodal_assets import ASSET_FIELDS, enrich_chunk_metadata

from embeddings.embedding_model import BgeM3EmbeddingModel, EmbeddingModelSettings
from ingest_data import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    HF_CACHE_DIR,
    SPARSE_VECTOR_NAME,
    _ensure_local_model,
    _stable_chunk_id,
    parse_sources,
)
from ingestion.entity_metadata import enrich_records_with_cross_references
from ingestion.parent_child import attach_parent_context


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
QDRANT_PATH = PROJECT_ROOT / "qdrant_db"
VISUAL_CAPTION_CACHE_DIR = PROJECT_ROOT / "data_cache" / "visual_captions"
BGE_MODEL_ID = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_LOCAL_DIR = PROJECT_ROOT / "hf_models_v2" / "bge-m3"
UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "32"))

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def create_fresh_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        logger.warning("Deleting existing Qdrant collection: %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    try:
        sparse_params = models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True),
            modifier=models.Modifier.IDF,
        )
    except Exception:
        sparse_params = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: sparse_params},
    )
    logger.info("Created Qdrant collection %s at %s", COLLECTION_NAME, QDRANT_PATH)


def load_bge_m3() -> BgeM3EmbeddingModel:
    logger.info("Initializing Google Gemini embedding model client: 'gemini-embedding-2'")
    return BgeM3EmbeddingModel(
        EmbeddingModelSettings(
            model_name_or_path="gemini-embedding-2",
            device="cpu",
            batch_size=100,
            max_sequence_length=2048,
            embedding_dimension=DENSE_VECTOR_SIZE,
            normalize_embeddings=True,
            cache_folder=HF_CACHE_DIR,
        )
    )


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _verified_chart_image_path(metadata: dict[str, Any]) -> Path:
    image_path = _safe_text(metadata.get("image_path"))
    if not image_path:
        raise ValueError(f"Chart visual chunk missing metadata['image_path']; chunk_id={metadata.get('chunk_id')}")
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Chart visual chunk image_path does not exist on disk: {image_path} "
            f"(resolved={path}) chunk_id={metadata.get('chunk_id')}"
        )
    return path


def _requires_visual_image_path(metadata: dict[str, Any]) -> bool:
    if metadata.get("contains_chart"):
        return True
    if metadata.get("contains_table") and metadata.get("document_type") == "pdf_visual":
        return True
    if metadata.get("contains_table") and metadata.get("content_type") == "visual":
        return True
    return False


def load_text_and_csv_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    logger.info("Loading existing text/PDF/CSV chunks from %s", ", ".join(str(path) for path in paths))
    records = parse_sources(paths, enrich_pdf_visuals=False)
    cleaned: list[dict[str, Any]] = []
    for record in records:
        text = _safe_text(record.get("text"))
        if not text:
            continue
        metadata = enrich_chunk_metadata(dict(record.get("metadata") or {}), text)
        metadata.setdefault("document_type", "text")
        metadata.setdefault("contains_chart", False)
        metadata.setdefault("contains_table", metadata.get("document_type") == "csv")
        metadata.setdefault("source", record.get("source", "unknown"))
        cleaned.append(
            {
                "text": text,
                "source": _safe_text(record.get("source")) or "unknown",
                "metadata": metadata,
            }
        )
    logger.info("Prepared %s text/PDF/CSV records", len(cleaned))
    return cleaned


def _visual_source_file(image_name: str, image_path: str) -> str:
    if image_name:
        stem = Path(image_name).stem
        marker = "_figure_"
        if marker in stem:
            return f"{stem.split(marker, 1)[0]}.pdf"
    if image_path:
        stem = Path(image_path).stem
        marker = "_figure_"
        if marker in stem:
            return f"{stem.split(marker, 1)[0]}.pdf"
    return "visual_caption_cache"


def load_visual_caption_records(cache_dir: Path = VISUAL_CAPTION_CACHE_DIR) -> list[dict[str, Any]]:
    if not cache_dir.exists():
        logger.warning("Visual caption cache directory does not exist: %s", cache_dir)
        return []

    records: list[dict[str, Any]] = []
    cache_files = sorted(cache_dir.glob("*.json"))
    logger.info("Loading %s cached visual caption file(s) from %s", len(cache_files), cache_dir)

    for index, cache_file in enumerate(cache_files, start=1):
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping unreadable visual caption cache %s: %s", cache_file, exc)
            continue

        structured_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        structured_metadata = (
            structured_payload.get("metadata")
            if isinstance(structured_payload.get("metadata"), dict)
            else {}
        )

        caption = _safe_text(structured_payload.get("text")) or _safe_text(payload.get("caption"))
        if not caption:
            logger.warning("Skipping empty visual caption cache: %s", cache_file)
            continue

        image_name = (
            _safe_text(structured_metadata.get("file_name"))
            or _safe_text(payload.get("image_name"))
            or Path(_safe_text(payload.get("image_path"))).name
        )
        image_path = _safe_text(payload.get("image_path"))
        image_hash = _safe_text(payload.get("image_hash")) or cache_file.stem
        figure_id = _safe_text(structured_metadata.get("figure_id")) or _safe_text(payload.get("figure_id")) or "Unknown"
        source_file = _visual_source_file(image_name, image_path)
        text = (
            "[VISUAL FIGURE DESCRIPTION]\n"
            f"Source file: {source_file}\n"
            f"Image name: {image_name}\n"
            f"Image hash: {image_hash}\n"
            f"Figure ID: {figure_id}\n\n"
            f"{caption}\n"
            "[/VISUAL FIGURE DESCRIPTION]"
        )

        records.append(
            {
                "text": text,
                "source": source_file,
                "metadata": {
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
                    "vision_model": _safe_text(payload.get("model")),
                    "contains_chart": True,
                    "contains_table": False,
                    "contains_diagram": True,
                    "visual_caption_index": index,
                },
            }
        )

    logger.info("Prepared %s cached visual caption records", len(records))
    return records


def build_integrated_records(data_paths: Iterable[Path]) -> list[dict[str, Any]]:
    text_records = parse_sources(data_paths, enrich_pdf_visuals=True)
    records = attach_parent_context(enrich_records_with_cross_references(text_records))
    logger.info(
        "Integrated deployment set: %s total chunks from structured ingestion",
        len(records),
        
    )
    return records


def _empty_sparse_vector() -> models.SparseVector:
    return models.SparseVector(indices=[], values=[])


def _point_id_for_chunk(chunk_id: str) -> str:
    stable = _safe_text(chunk_id) or str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable))


def build_points(records: list[dict[str, Any]], dense_vectors: list[list[float]]) -> list[models.PointStruct]:
    if len(records) != len(dense_vectors):
        raise ValueError(f"Record/vector count mismatch: {len(records)} records vs {len(dense_vectors)} vectors")

    points: list[models.PointStruct] = []
    for record, dense_vector in zip(records, dense_vectors):
        text = _safe_text(record.get("text"))
        metadata = dict(record.get("metadata") or {})
        chunk_id = _safe_text(metadata.get("chunk_id")) or _stable_chunk_id(text)

        if len(dense_vector) != DENSE_VECTOR_SIZE:
            raise ValueError(f"Expected BGE-M3 vector dimension {DENSE_VECTOR_SIZE}, got {len(dense_vector)}")

        if _requires_visual_image_path(metadata):
            _verified_chart_image_path(metadata)
            print(
                f"VALIDATION [Qdrant Payload]: chunk_id={chunk_id} "
                f"contains_chart={metadata.get('contains_chart')} contains_table={metadata.get('contains_table')} "
                f"image_path={metadata.get('image_path')} exists=True",
                flush=True,
            )

        payload = {
            "text": text,
            "page_content": text,
            "source": _safe_text(record.get("source")) or _safe_text(metadata.get("source")) or "unknown",
            "image_path": metadata.get("image_path"),
            "contains_chart": bool(metadata.get("contains_chart")),
            "contains_table": bool(metadata.get("contains_table")),
            "contains_figure": bool(metadata.get("contains_figure")),
            "contains_image": bool(metadata.get("contains_image")),
            "contains_csv": bool(metadata.get("contains_csv")),
            "metadata": metadata,
        }
        for key in ASSET_FIELDS:
            if metadata.get(key) not in ("", None, [], {}):
                payload[key] = metadata[key]
        if not _safe_text(payload["text"]):
            raise ValueError(f"Cannot upsert record without root payload['text']; chunk_id={chunk_id}")
        points.append(
            models.PointStruct(
                id=_point_id_for_chunk(chunk_id),
                vector={
                    DENSE_VECTOR_NAME: [float(value) for value in dense_vector],
                    SPARSE_VECTOR_NAME: _empty_sparse_vector(),
                },
                payload=payload,
            )
        )
    return points


def upsert_points(client: QdrantClient, points: list[models.PointStruct]) -> int:
    uploaded = 0
    total = len(points)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = points[start : start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded += len(batch)
        logger.info("Upserted chunk %s/%s...", uploaded, total)
    return uploaded


def deploy(data_paths: Iterable[Path], recreate_collection: bool = True) -> int:
    records = build_integrated_records(data_paths)
    if not records:
        logger.warning("No records found for deployment.")
        return 0

    client = get_qdrant_client()
    try:
        if recreate_collection:
            create_fresh_collection(client)
        elif not client.collection_exists(COLLECTION_NAME):
            create_fresh_collection(client)

        embedder = load_bge_m3()
        texts = [record["text"] for record in records]
        logger.info("Embedding %s integrated chunks with BGE-M3", len(texts))
        dense_vectors = embedder.embed_documents(texts, batch_size=EMBEDDING_BATCH_SIZE)

        points = build_points(records, dense_vectors)
        uploaded = upsert_points(client, points)
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        logger.info("Deployment complete. Upserted %s points. Qdrant exact count: %s", uploaded, count)
        return uploaded
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy integrated PDF/CSV/visual-caption chunks into local Qdrant.")
    parser.add_argument("sources", nargs="*", default=[str(DATA_DIR)], help="Files or directories to parse for text/CSV/PDF chunks.")
    parser.add_argument("--preserve", action="store_true", help="Do not recreate the Qdrant collection before deployment.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    source_paths = [Path(source).expanduser().resolve() for source in args.sources]
    deploy(source_paths, recreate_collection=not args.preserve)


if __name__ == "__main__":
    main()
