from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUMBA_CACHE_DIR = PROJECT_ROOT / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
LOCAL_HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
LOCAL_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(LOCAL_HF_CACHE_DIR / "hub"))
TESSERACT_DIR = Path(os.getenv("TESSERACT_DIR", r"C:\Program Files\Tesseract-OCR"))
if (TESSERACT_DIR / "tesseract.exe").exists():
    os.environ["PATH"] = f"{TESSERACT_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from embeddings.embedding_model import BgeM3EmbeddingModel, EmbeddingModelSettings
from ingest_data import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    DOCLING_ARTIFACTS_PATH,
    DOCLING_OCR_ENGINE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    HF_CACHE_DIR,
    SPARSE_VECTOR_NAME,
    _ensure_local_model,
)


load_dotenv()

DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_CROP_DIR = PROJECT_ROOT / "assets" / "extracted_images"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "assets" / "extracted_tables"
DEFAULT_PROGRESS_PATH = PROJECT_ROOT / "ingestion_progress.json"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = PROJECT_ROOT / "qdrant_db"
BGE_MODEL_ID = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_LOCAL_DIR = PROJECT_ROOT / "hf_models_v2" / "bge-m3"
UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "32"))
PDF_PAGE_WINDOW_SIZE = max(1, int(os.getenv("PDF_PAGE_WINDOW_SIZE", "8")))

ENTITY_PATTERN = re.compile(
    r"\b(?P<kind>Figure|Fig\.?|Chart|Diagram|Table)\s*(?P<number>[A-Za-z]?\d+(?:\.\d+)*)\b",
    flags=re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# Optional fallback parser install command:
#   pip install "unstructured[pdf]"


@dataclass
class AssetBinding:
    entity_id: str
    entity_kind: str
    page_no: int
    image_path: str = ""
    csv_path: str = ""
    caption: str = ""
    bbox: dict[str, Any] | None = None


@dataclass
class PagePayload:
    page_no: int
    parent_text: str
    parent_metadata: dict[str, Any]
    child_records: list[dict[str, Any]] = field(default_factory=list)
    bindings: dict[str, AssetBinding] = field(default_factory=dict)


def get_qdrant_client(use_http: bool = True) -> QdrantClient:
    if use_http:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))


def load_embedding_model() -> BgeM3EmbeddingModel:
    model_path = _ensure_local_model(BGE_MODEL_ID, BGE_LOCAL_DIR)
    return BgeM3EmbeddingModel(
        EmbeddingModelSettings(
            model_name_or_path=model_path,
            device=os.getenv("BGE_M3_DEVICE", "cpu"),
            batch_size=EMBEDDING_BATCH_SIZE,
            max_sequence_length=EMBEDDING_MAX_LENGTH,
            embedding_dimension=DENSE_VECTOR_SIZE,
            normalize_embeddings=True,
            cache_folder=HF_CACHE_DIR,
        )
    )


def recreate_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        logger.warning("Deleting existing Qdrant collection before synchronized reindex: %s", collection_name)
        client.delete_collection(collection_name)

    try:
        sparse_params = models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True),
            modifier=models.Modifier.IDF,
        )
    except Exception:
        sparse_params = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: sparse_params},
    )
    for field_name, schema in {
        "source": models.PayloadSchemaType.KEYWORD,
        "metadata.document_type": models.PayloadSchemaType.KEYWORD,
        "metadata.entity_id": models.PayloadSchemaType.KEYWORD,
        "metadata.entity_ids": models.PayloadSchemaType.KEYWORD,
        "metadata.page_no": models.PayloadSchemaType.INTEGER,
        "metadata.parent_id": models.PayloadSchemaType.KEYWORD,
        "metadata.contains_chart": models.PayloadSchemaType.BOOL,
        "metadata.contains_table": models.PayloadSchemaType.BOOL,
        "metadata.image_path": models.PayloadSchemaType.KEYWORD,
        "metadata.csv_path": models.PayloadSchemaType.KEYWORD,
        "text": models.PayloadSchemaType.TEXT,
        "page_content": models.PayloadSchemaType.TEXT,
    }.items():
        try:
            client.create_payload_index(collection_name=collection_name, field_name=field_name, field_schema=schema)
        except Exception as exc:
            logger.debug("Payload index skipped for %s: %s", field_name, exc)


def load_progress(progress_path: Path = DEFAULT_PROGRESS_PATH) -> dict[str, Any]:
    if not progress_path.exists():
        return {"pdf_pages": {}, "csv_files": []}
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read progress file %s; starting fresh progress state: %s", progress_path, exc)
        return {"pdf_pages": {}, "csv_files": []}
    data.setdefault("pdf_pages", {})
    data.setdefault("csv_files", [])
    return data


def save_progress(progress: dict[str, Any], progress_path: Path = DEFAULT_PROGRESS_PATH) -> None:
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")


def completed_pages(progress: dict[str, Any], pdf_path: Path) -> set[int]:
    return {int(page) for page in progress.get("pdf_pages", {}).get(str(pdf_path.resolve()), [])}


def mark_page_completed(progress: dict[str, Any], pdf_path: Path, page_no: int, progress_path: Path) -> None:
    key = str(pdf_path.resolve())
    pages = {int(page) for page in progress.setdefault("pdf_pages", {}).get(key, [])}
    pages.add(int(page_no))
    progress["pdf_pages"][key] = sorted(pages)
    save_progress(progress, progress_path)


def csv_completed(progress: dict[str, Any], csv_path: Path) -> bool:
    return str(csv_path.resolve()) in set(progress.get("csv_files", []))


def mark_csv_completed(progress: dict[str, Any], csv_path: Path, progress_path: Path) -> None:
    files = set(progress.setdefault("csv_files", []))
    files.add(str(csv_path.resolve()))
    progress["csv_files"] = sorted(files)
    save_progress(progress, progress_path)


class PageVisualCropper:
    """Synchronous page-scoped visual crop adapter.

    It uses Unstructured's image extraction if available and renames each saved
    crop to the engine's deterministic entity path.
    """

    def extract_page(
        self,
        *,
        pdf_path: Path,
        page_no: int,
        bindings: dict[str, AssetBinding],
        output_dir: Path,
    ) -> dict[str, AssetBinding]:
        if not bindings or all(binding.image_path or binding.entity_kind == "table" for binding in bindings.values()):
            return bindings
        output_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("tesseract") is None:
            logger.warning(
                "Skipping Unstructured visual crop fallback for page %s because tesseract is not installed.",
                page_no,
            )
            return bindings

        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            logger.warning("unstructured[pdf] is unavailable; cannot crop visuals for page %s", page_no)
            return bindings

        page_tmp_dir = output_dir / "_tmp" / f"page_{page_no}"
        page_tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=os.getenv("PDF_VISUAL_STRATEGY", "hi_res"),
                infer_table_structure=False,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=str(page_tmp_dir),
                starting_page_number=page_no,
                ending_page_number=page_no,
            )
        except Exception as exc:
            logger.warning("Visual crop extraction failed for page %s; continuing text ingestion: %s", page_no, exc)
            return bindings

        saved_paths = _extracted_image_paths(elements)
        pending = [binding for binding in bindings.values() if binding.entity_kind != "table" and not binding.image_path]
        for index, binding in enumerate(pending):
            source_path = saved_paths[index] if index < len(saved_paths) else None
            if not source_path or not source_path.exists():
                logger.warning(
                    "No crop file returned for %s on page %s. image_path remains unset until fixed.",
                    binding.entity_id,
                    page_no,
                )
                continue
            final_path = output_dir / f"page_{page_no}_{_path_token(binding.entity_id)}.png"
            source_path.replace(final_path)
            binding.image_path = str(final_path.resolve())
            logger.info("Saved synchronized crop: entity=%s path=%s", binding.entity_id, binding.image_path)

        return bindings


class SynchronizedMultimodalEngine:
    def __init__(
        self,
        *,
        client: QdrantClient,
        embedding_model: BgeM3EmbeddingModel,
        collection_name: str = COLLECTION_NAME,
        crop_dir: Path = DEFAULT_CROP_DIR,
        table_dir: Path = DEFAULT_TABLE_DIR,
        progress_path: Path = DEFAULT_PROGRESS_PATH,
    ) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.cropper = PageVisualCropper()
        self.crop_dir = crop_dir
        self.table_dir = table_dir
        self.progress_path = progress_path
        self.progress = load_progress(progress_path)

    def run(self, input_paths: Iterable[Path], *, recreate: bool = True) -> int:
        has_progress = bool(self.progress.get("pdf_pages") or self.progress.get("csv_files"))
        if recreate and not has_progress:
            recreate_collection(self.client, self.collection_name)
        elif recreate and has_progress:
            logger.info("Progress file exists at %s; resume mode will not recreate Qdrant.", self.progress_path)

        total = 0
        for path in input_paths:
            if path.suffix.lower() == ".pdf":
                total += self.ingest_pdf(path)
            elif path.suffix.lower() == ".csv":
                total += self.ingest_csv(path)
            else:
                logger.info("Skipping unsupported file: %s", path)
        return total

    def ingest_pdf(self, pdf_path: Path) -> int:
        logger.info("Docling synchronized PDF ingestion started: %s", pdf_path)
        uploaded = 0
        already_done = completed_pages(self.progress, pdf_path)
        physical_pages = _pdf_page_count(pdf_path)
        unfinished_pages = [page_no for page_no in range(1, physical_pages + 1) if page_no not in already_done]
        if not unfinished_pages:
            logger.info("All PDF pages already completed by checkpoint: %s", pdf_path.name)
            return 0

        logger.info(
            "Resume windowing enabled: %s unfinished pages across %s physical pages; window=%s",
            len(unfinished_pages),
            physical_pages,
            PDF_PAGE_WINDOW_SIZE,
        )

        for range_start, range_end in _page_windows(unfinished_pages, PDF_PAGE_WINDOW_SIZE):
            logger.info("Converting Docling page range %s-%s for %s", range_start, range_end, pdf_path.name)
            document = None
            page_payloads: dict[int, PagePayload] = {}
            conversion_failed = False
            try:
                document = _convert_pdf_with_docling(pdf_path, page_range=(range_start, range_end))
                page_payloads = _docling_pages_to_payloads(document, pdf_path, self.crop_dir)
            except Exception as exc:
                conversion_failed = True
                logger.warning(
                    "Docling range conversion failed for %s pages %s-%s: %s",
                    pdf_path.name,
                    range_start,
                    range_end,
                    exc,
                )

            if not page_payloads:
                logger.warning("No Docling payloads produced for %s pages %s-%s", pdf_path.name, range_start, range_end)
            if not conversion_failed:
                for empty_page_no in range(range_start, range_end + 1):
                    if empty_page_no not in page_payloads and empty_page_no not in completed_pages(self.progress, pdf_path):
                        logger.info(
                            "Marking page as completed with no text payload: pdf=%s page=%s",
                            pdf_path.name,
                            empty_page_no,
                        )
                        mark_page_completed(self.progress, pdf_path, empty_page_no, self.progress_path)

            for page_no in sorted(page_payloads):
                if page_no in completed_pages(self.progress, pdf_path):
                    logger.info("Skipping completed page from checkpoint: pdf=%s page=%s", pdf_path.name, page_no)
                    continue
                page_payload = page_payloads[page_no]
                docling_page_data = page_payload
                image_crops = None
                try:
                    image_crops = self.cropper.extract_page(
                        pdf_path=pdf_path,
                        page_no=page_no,
                        bindings=docling_page_data.bindings,
                        output_dir=self.crop_dir,
                    )
                    docling_page_data.bindings.update(image_crops)
                    self._save_table_csvs(docling_page_data, pdf_path)
                    self._propagate_bindings(docling_page_data)
                    records = self._page_records(docling_page_data)
                    self._validate_records(records)
                    uploaded += self._upsert_records(records)
                    mark_page_completed(self.progress, pdf_path, page_no, self.progress_path)
                finally:
                    try:
                        del records
                    except UnboundLocalError:
                        pass
                    del docling_page_data, image_crops
                    gc.collect()
            del document, page_payloads
            gc.collect()
        gc.collect()
        return uploaded

    def ingest_csv(self, csv_path: Path) -> int:
        if csv_completed(self.progress, csv_path):
            logger.info("Skipping completed CSV from checkpoint: %s", csv_path)
            return 0
        records: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=1):
                text = " | ".join(f"{key}: {value}" for key, value in row.items())
                records.append(
                    {
                        "text": text,
                        "source": csv_path.name,
                        "metadata": {
                            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{csv_path}|{row_index}|{text}")),
                            "document_type": "csv",
                            "source_file": csv_path.name,
                            "source_path": str(csv_path.resolve()),
                            "row_id": row_index,
                            "contains_table": True,
                            "contains_chart": False,
                        },
                    }
                )
        self._validate_records(records)
        uploaded = self._upsert_records(records)
        mark_csv_completed(self.progress, csv_path, self.progress_path)
        del records
        gc.collect()
        return uploaded

    def _save_table_csvs(self, page_payload: PagePayload, pdf_path: Path) -> None:
        self.table_dir.mkdir(parents=True, exist_ok=True)
        for binding in page_payload.bindings.values():
            if binding.entity_kind != "table":
                continue
            csv_path = self.table_dir / f"page_{page_payload.page_no}_{_path_token(binding.entity_id)}.csv"
            rows = _markdown_table_rows(_text_for_entity(page_payload.child_records, binding.entity_id))
            if not rows:
                rows = [["entity_id", "page_no", "context"], [binding.entity_id, str(page_payload.page_no), binding.caption]]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)
            binding.csv_path = str(csv_path.resolve())
            logger.info("Saved synchronized table CSV: entity=%s path=%s", binding.entity_id, binding.csv_path)

    def _propagate_bindings(self, page_payload: PagePayload) -> None:
        page_entity_ids = sorted(page_payload.bindings)
        page_payload.parent_metadata["entity_ids"] = page_entity_ids

        for binding in page_payload.bindings.values():
            if binding.image_path:
                page_payload.parent_metadata["contains_chart"] = True
                page_payload.parent_metadata.setdefault("image_paths", {})[binding.entity_id] = binding.image_path
                if not page_payload.parent_metadata.get("image_path"):
                    page_payload.parent_metadata["image_path"] = binding.image_path
            if binding.csv_path:
                page_payload.parent_metadata["contains_table"] = True
                page_payload.parent_metadata.setdefault("csv_paths", {})[binding.entity_id] = binding.csv_path
                if not page_payload.parent_metadata.get("csv_path"):
                    page_payload.parent_metadata["csv_path"] = binding.csv_path

        for record in page_payload.child_records:
            metadata = record["metadata"]
            text_blob = f"{record['text']} {metadata.get('entity_id', '')}"
            matched = [binding for binding in page_payload.bindings.values() if _entity_matches_text(binding.entity_id, text_blob)]
            if not matched and len(page_payload.bindings) == 1:
                matched = list(page_payload.bindings.values())
            for binding in matched:
                metadata["entity_id"] = binding.entity_id
                if binding.image_path:
                    metadata["contains_chart"] = True
                    metadata["image_path"] = binding.image_path
                if binding.csv_path:
                    metadata["contains_table"] = True
                    metadata["csv_path"] = binding.csv_path

    def _page_records(self, page_payload: PagePayload) -> list[dict[str, Any]]:
        parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(page_payload.parent_metadata)))
        parent_id = f"parent-{parent_uuid}"
        parent_record = {
            "text": page_payload.parent_text,
            "source": page_payload.parent_metadata["source_file"],
            "metadata": {
                **page_payload.parent_metadata,
                "chunk_id": parent_uuid,
                "parent_id": parent_id,
                "chunk_role": "parent",
                "parent_text": page_payload.parent_text,
            },
        }
        records = [parent_record]
        for index, child in enumerate(page_payload.child_records):
            metadata = {
                **page_payload.parent_metadata,
                **child["metadata"],
                "parent_id": parent_id,
                "chunk_role": "child",
                "parent_text": page_payload.parent_text,
            }
            text = child["text"]
            metadata["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{metadata.get('source_file')}|{index}|{text}"))
            records.append({"text": text, "source": metadata["source_file"], "metadata": metadata})
        return records

    def _validate_records(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            metadata = record.get("metadata") or {}
            image_path = str(metadata.get("image_path") or "").strip()
            if metadata.get("contains_chart"):
                fixed = _fix_existing_path(image_path, self.crop_dir)
                if not fixed:
                    logger.warning(
                        "Visual chunk has contains_chart=True but no valid image_path; treating as text-only. "
                        "chunk_id=%s entity_id=%s bad_path=%s",
                        metadata.get("chunk_id"),
                        metadata.get("entity_id"),
                        image_path,
                    )
                    metadata["contains_chart"] = False
                    metadata.pop("image_path", None)
                else:
                    metadata["image_path"] = fixed
                    logger.info("VALIDATED image_path: chunk_id=%s path=%s", metadata.get("chunk_id"), fixed)
            csv_path = str(metadata.get("csv_path") or "").strip()
            if metadata.get("contains_table") and csv_path:
                fixed_csv = _fix_existing_path(csv_path, self.table_dir)
                if fixed_csv:
                    metadata["csv_path"] = fixed_csv

    def _upsert_records(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        texts = [record["text"] for record in records]
        vectors = self.embedding_model.embed_documents(texts, batch_size=EMBEDDING_BATCH_SIZE)
        points = []
        for record, vector in zip(records, vectors):
            metadata = dict(record.get("metadata") or {})
            payload = {
                "text": record["text"],
                "page_content": record["text"],
                "source": record.get("source") or metadata.get("source_file") or "unknown",
                "contains_chart": bool(metadata.get("contains_chart")),
                "contains_table": bool(metadata.get("contains_table")),
                "metadata": metadata,
            }
            if metadata.get("image_path"):
                payload["image_path"] = metadata["image_path"]
            if metadata.get("csv_path"):
                payload["csv_path"] = metadata["csv_path"]
            points.append(
                models.PointStruct(
                    id=metadata["chunk_id"],
                    vector={
                        DENSE_VECTOR_NAME: [float(value) for value in vector],
                        SPARSE_VECTOR_NAME: models.SparseVector(indices=[], values=[]),
                    },
                    payload=payload,
                )
            )

        uploaded = 0
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
            uploaded += len(batch)
            logger.info("Synchronized upsert complete: %s/%s records", uploaded, len(points))
        return uploaded


def _docling_pages_to_payloads(document: Any, pdf_path: Path, crop_dir: Path) -> dict[int, PagePayload]:
    pages: dict[int, PagePayload] = {}
    for item, _level in document.iterate_items():
        page_no = _item_page_no(item)
        if page_no is None:
            continue
        text = _item_text(item, document)
        if not text:
            continue
        page = pages.setdefault(
            page_no,
            PagePayload(
                page_no=page_no,
                parent_text="",
                parent_metadata={
                    "document_type": "pdf",
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path.resolve()),
                    "page_no": page_no,
                    "docling_page_no": page_no,
                    "contains_chart": False,
                    "contains_table": False,
                },
            ),
        )
        entity = _extract_entity(text)
        item_metadata = {
            "document_type": "pdf",
            "source_file": pdf_path.name,
            "source_path": str(pdf_path.resolve()),
            "page_no": page_no,
            "docling_page_no": page_no,
            "docling_label": str(getattr(item, "label", "") or ""),
            "docling_self_ref": str(getattr(item, "self_ref", "") or ""),
            "contains_chart": False,
            "contains_table": False,
        }
        if entity:
            entity_id, kind = entity
            item_metadata["entity_id"] = entity_id
            binding = page.bindings.setdefault(
                entity_id,
                AssetBinding(entity_id=entity_id, entity_kind=kind, page_no=page_no, caption=text[:500]),
            )
            if kind == "table":
                item_metadata["contains_table"] = True
            else:
                item_metadata["contains_chart"] = True
                binding.bbox = _item_bbox(item)
        elif _is_docling_picture_item(item):
            entity_id = f"Figure_page_{page_no}"
            item_metadata["entity_id"] = entity_id
            item_metadata["contains_chart"] = True
            binding = page.bindings.setdefault(
                entity_id,
                AssetBinding(entity_id=entity_id, entity_kind="figure", page_no=page_no, caption=text[:500], bbox=_item_bbox(item)),
            )
            if not binding.image_path:
                binding.image_path = _save_docling_picture_image(item, document, crop_dir, page_no, entity_id)
        page.child_records.append({"text": text, "metadata": item_metadata})

    for page in pages.values():
        page.parent_text = "\n\n".join(record["text"] for record in page.child_records).strip()
    return pages


def _pdf_page_count(pdf_path: Path) -> int:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def _page_windows(page_numbers: list[int], window_size: int) -> Iterable[tuple[int, int]]:
    if not page_numbers:
        return
    group_start = page_numbers[0]
    previous = page_numbers[0]
    group: list[int] = [page_numbers[0]]
    for page_no in page_numbers[1:]:
        if page_no == previous + 1 and len(group) < window_size:
            group.append(page_no)
        else:
            yield group_start, previous
            group_start = page_no
            group = [page_no]
        previous = page_no
    yield group_start, previous


def _convert_pdf_with_docling(pdf_path: Path, page_range: tuple[int, int] | None = None) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, OcrEngine, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    DOCLING_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(DOCLING_ARTIFACTS_PATH))

    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = DOCLING_ARTIFACTS_PATH
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = False
    pipeline_options.images_scale = 2.0
    pipeline_options.do_chart_extraction = False
    pipeline_options.do_picture_classification = False
    pipeline_options.do_picture_description = False
    pipeline_options.do_table_structure = os.getenv("DOCLING_TABLE_STRUCTURE", "false").lower() in {"1", "true", "yes", "on"}

    if DOCLING_OCR_ENGINE in {"off", "false", "0", "none", "disabled"}:
        pipeline_options.do_ocr = False
    elif DOCLING_OCR_ENGINE == OcrEngine.EASYOCR.value:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=True,
            model_storage_directory=str(DOCLING_ARTIFACTS_PATH / "easyocr"),
        )

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    if page_range:
        return converter.convert(str(pdf_path), page_range=page_range).document
    return converter.convert(str(pdf_path)).document


def _save_docling_picture_image(item: Any, document: Any, crop_dir: Path, page_no: int, entity_id: str) -> str:
    image_getter = getattr(item, "get_image", None)
    if not callable(image_getter):
        return ""
    try:
        image = image_getter(document)
    except Exception as exc:
        logger.warning("Docling get_image failed for %s on page %s: %s", entity_id, page_no, exc)
        return ""
    if image is None:
        return ""
    crop_dir.mkdir(parents=True, exist_ok=True)
    image_path = crop_dir / f"page_{page_no}_{_path_token(entity_id)}.png"
    try:
        image.save(image_path, "PNG")
    except Exception as exc:
        logger.warning("Could not save Docling picture crop for %s to %s: %s", entity_id, image_path, exc)
        return ""
    resolved = str(image_path.resolve())
    logger.info("Saved Docling picture crop: entity=%s path=%s", entity_id, resolved)
    return resolved


def _extract_entity(text: str) -> tuple[str, str] | None:
    match = ENTITY_PATTERN.search(text or "")
    if not match:
        return None
    kind_raw = match.group("kind").lower().rstrip(".")
    number = match.group("number")
    kind = "table" if kind_raw == "table" else "figure"
    label = "Table" if kind == "table" else "Figure"
    return f"{label}_{number}", kind


def _entity_matches_text(entity_id: str, text: str) -> bool:
    normalized_entity = re.sub(r"[^a-z0-9]+", "", entity_id.lower())
    normalized_text = re.sub(r"[^a-z0-9]+", "", str(text or "").lower())
    return normalized_entity in normalized_text


def _item_page_no(item: Any) -> int | None:
    prov = list(getattr(item, "prov", []) or [])
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def _item_text(item: Any, document: Any) -> str:
    if hasattr(item, "text") and getattr(item, "text"):
        return str(getattr(item, "text")).strip()
    caption_text = getattr(item, "caption_text", None)
    if callable(caption_text):
        try:
            return str(caption_text(document) or "").strip()
        except TypeError:
            return str(caption_text() or "").strip()
        except Exception:
            return ""
    return ""


def _item_bbox(item: Any) -> dict[str, Any] | None:
    prov = list(getattr(item, "prov", []) or [])
    bbox = getattr(prov[0], "bbox", None) if prov else None
    if bbox is None:
        return None
    return {key: getattr(bbox, key) for key in ("l", "t", "r", "b", "coord_origin") if hasattr(bbox, key)}


def _is_docling_picture_item(item: Any) -> bool:
    label = str(getattr(item, "label", "") or "").lower()
    type_name = type(item).__name__.lower()
    return "picture" in label or "picture" in type_name or "image" in label


def _extracted_image_paths(elements: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for element in elements:
        metadata = getattr(element, "metadata", None)
        image_path = _metadata_value(metadata, "image_path")
        if image_path:
            paths.append(Path(str(image_path)))
    return paths


def _metadata_value(metadata: Any, key: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _markdown_table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and not all(set(cell) <= {"-", ":"} for cell in cells):
            rows.append(cells)
    return rows


def _text_for_entity(records: list[dict[str, Any]], entity_id: str) -> str:
    parts = [record["text"] for record in records if _entity_matches_text(entity_id, record["text"])]
    return "\n\n".join(parts)


def _fix_existing_path(path_string: str, base_dir: Path) -> str:
    if not path_string:
        return ""
    path = Path(path_string)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, base_dir / path.name])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", str(value or "")).strip("_") or "asset"


def _iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".pdf", ".csv"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant with synchronized Docling/Vision/CSV bindings.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_DATA_DIR), help="PDF/CSV file or directory")
    parser.add_argument("--local-qdrant", action="store_true", help="Use local qdrant_db instead of QDRANT_URL")
    parser.add_argument("--no-recreate", action="store_true", help="Do not delete/recreate the Qdrant collection")
    parser.add_argument("--progress-file", default=str(DEFAULT_PROGRESS_PATH), help="JSON checkpoint file path")
    parser.add_argument("--fresh", action="store_true", help="Delete the progress file and recreate Qdrant from page 1")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    progress_path = Path(args.progress_file)
    if args.fresh and progress_path.exists():
        logger.warning("Deleting progress checkpoint for fresh run: %s", progress_path)
        progress_path.unlink()
    input_files = _iter_input_files(Path(args.input))
    client = get_qdrant_client(use_http=not args.local_qdrant)
    engine = SynchronizedMultimodalEngine(
        client=client,
        embedding_model=load_embedding_model(),
        progress_path=progress_path,
    )
    uploaded = engine.run(input_files, recreate=(not args.no_recreate) or args.fresh)
    logger.info("Synchronized reindex finished. Uploaded %s records.", uploaded)


if __name__ == "__main__":
    main()
