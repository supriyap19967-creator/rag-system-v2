from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


load_dotenv()

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = Path(os.getenv("HF_HOME", str(Path(CURRENT_FILE_DIR) / "hf_cache_v2")))
HF_HUB_CACHE_DIR = Path(os.getenv("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub")))
TRANSFORMERS_CACHE_DIR = Path(os.getenv("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers")))
LOCAL_MODELS_DIR = Path(os.getenv("LOCAL_MODELS_DIR", str(Path(CURRENT_FILE_DIR) / "hf_models_v2")))
BGE_MODEL_ID = "BAAI/bge-m3"
BGE_LOCAL_DIR = LOCAL_MODELS_DIR / "bge-m3"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE_DIR))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from embeddings.embedding_model import BgeM3EmbeddingModel, EmbeddingModelSettings
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient, models
from vectordb.qdrant_client_manager import get_qdrant_client as build_managed_qdrant_client
from app.multimodal_assets import ASSET_FIELDS, enrich_chunk_metadata, validate_asset_path
from ingestion.gemini_vision_caption import GeminiVisionCaptioner
from ingestion.entity_metadata import enrich_records_with_cross_references
from ingestion.parent_child import attach_parent_context
from ingestion.pipeline import MultimodalIngestionPipeline
from ingestion.schemas import ExtractedImage


QDRANT_PATH = os.path.join(CURRENT_FILE_DIR, "qdrant_db")
COLLECTION_NAME = "conversational_rag"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 384
MAX_PARAGRAPH_TOKENS = 768
EMBEDDING_MAX_LENGTH = 1024
EMBEDDING_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 64
EXTRACTED_CHARTS_DIR = Path(CURRENT_FILE_DIR) / "extracted_images"
DOCLING_ARTIFACTS_PATH = Path(os.getenv("DOCLING_ARTIFACTS_PATH", str(Path(CURRENT_FILE_DIR) / "docling_models")))
DOCLING_OCR_ENGINE = os.getenv("DOCLING_OCR_ENGINE", "off").strip().lower()

logger = logging.getLogger(__name__)


def _sanitize_image_token(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "visual"


def _entity_id_from_label(label: str, fallback_index: int) -> str:
    match = re.search(
        r"\b(?:fig(?:ure)?|chart|diagram|table)\s*([A-Za-z]?\d+(?:\.\d+)*)\b",
        str(label or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return f"figure_{match.group(1).replace('.', '_').lower()}"
    return f"figure_{fallback_index:04d}"


def _verified_chart_image_path(metadata: dict[str, Any]) -> Path:
    image_path = str(metadata.get("image_path") or "").strip()
    if not image_path:
        raise ValueError(f"Chart/table chunk missing metadata['image_path']; chunk_id={metadata.get('chunk_id')}")
    path = Path(image_path)
    if not path.is_absolute():
        path = Path(CURRENT_FILE_DIR) / path
    if not path.exists():
        raise FileNotFoundError(
            f"Chart/table chunk image_path does not exist on disk: {metadata.get('image_path')} "
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


def _normalize_visual_metadata_paths(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coalesce typed visual paths and persist absolute on-disk paths into metadata."""

    normalized = dict(metadata)
    if not normalized.get("image_path"):
        for key in ("figure_image_path", "chart_image_path", "table_image_path", "diagram_image_path"):
            candidate = normalized.get(key)
            if candidate not in ("", None):
                normalized["image_path"] = candidate
                break

    path_keys = (
        "image_path",
        "figure_image_path",
        "chart_image_path",
        "table_image_path",
        "diagram_image_path",
    )
    list_keys = (
        "image_paths",
        "figure_image_paths",
        "chart_image_paths",
        "table_image_paths",
        "diagram_image_paths",
        "asset_paths",
    )
    for key in path_keys:
        value = normalized.get(key)
        if value in ("", None):
            continue
        validation = validate_asset_path(value)
        if validation.ok:
            normalized[key] = validation.path

    for key in list_keys:
        values = normalized.get(key)
        if not isinstance(values, list):
            continue
        resolved_values: list[str] = []
        for value in values:
            validation = validate_asset_path(value)
            if validation.ok:
                resolved_values.append(validation.path)
        if resolved_values:
            normalized[key] = resolved_values
            normalized.setdefault("image_path", resolved_values[0])

    return normalized


def get_qdrant_client() -> QdrantClient:
    return build_managed_qdrant_client()


@lru_cache(maxsize=1)
def get_embedding_model() -> BgeM3EmbeddingModel:
    return BgeM3EmbeddingModel(
        EmbeddingModelSettings(
            model_name_or_path=_ensure_local_model(BGE_MODEL_ID, BGE_LOCAL_DIR),
            device=os.getenv("BGE_M3_DEVICE", "cpu"),
            batch_size=EMBEDDING_BATCH_SIZE,
            max_sequence_length=EMBEDDING_MAX_LENGTH,
            embedding_dimension=DENSE_VECTOR_SIZE,
            normalize_embeddings=True,
            cache_folder=HF_CACHE_DIR,
        )
    )


@lru_cache(maxsize=1)
def get_vision_captioner() -> GeminiVisionCaptioner:
    return GeminiVisionCaptioner()


def _ensure_local_model(repo_id: str, local_dir: Path) -> str:
    """Download a model into a flat project-local folder to avoid Windows cache symlink failures."""

    if (local_dir / "config.json").exists():
        return str(local_dir)

    logger.info("Downloading %s into %s", repo_id, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        cache_dir=HF_CACHE_DIR,
        max_workers=2,
    )
    return str(local_dir)


def create_or_recreate_collection(client: QdrantClient, recreate: bool = False) -> None:
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and recreate:
        logger.warning("Recreating Qdrant collection %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if exists:
        logger.info("Qdrant collection %s already exists", COLLECTION_NAME)
        ensure_payload_indexes(client)
        return

    sparse_params = _sparse_vector_params_with_idf()
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
    logger.info("Created collection %s with named dense+sparse vectors", COLLECTION_NAME)
    ensure_payload_indexes(client)


def ensure_payload_indexes(client: QdrantClient) -> None:
    payload_indexes = {
        "source": models.PayloadSchemaType.KEYWORD,
        "metadata.document_type": models.PayloadSchemaType.KEYWORD,
        "metadata.source_file": models.PayloadSchemaType.KEYWORD,
        "metadata.source_path": models.PayloadSchemaType.KEYWORD,
        "metadata.chunk_id": models.PayloadSchemaType.KEYWORD,
        "metadata.row_id": models.PayloadSchemaType.INTEGER,
        "metadata.contains_table": models.PayloadSchemaType.BOOL,
        "metadata.contains_chart": models.PayloadSchemaType.BOOL,
        "metadata.entity_id": models.PayloadSchemaType.KEYWORD,
        "metadata.entity_ids": models.PayloadSchemaType.KEYWORD,
        "metadata.cross_reference": models.PayloadSchemaType.KEYWORD,
        "metadata.cross_references": models.PayloadSchemaType.KEYWORD,
        "metadata.parent_id": models.PayloadSchemaType.KEYWORD,
        "text": models.PayloadSchemaType.TEXT,
        "page_content": models.PayloadSchemaType.TEXT,
    }
    for field_name, field_schema in payload_indexes.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            logger.debug("Payload index %s skipped or already exists: %s", field_name, exc)


def _sparse_vector_params_with_idf() -> models.SparseVectorParams:
    try:
        return models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True),
            modifier=models.Modifier.IDF,
        )
    except Exception:
        logger.warning("Qdrant client does not expose sparse IDF/on-disk params; using default sparse params.")
        return models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))


def parse_sources(paths: Iterable[str | Path], enrich_pdf_visuals: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        if path.is_dir():
            records.extend(parse_sources(_iter_supported_files(path), enrich_pdf_visuals=enrich_pdf_visuals))
        elif path.suffix.lower() == ".pdf":
            result = MultimodalIngestionPipeline().ingest_sync(path)
            records.extend(
                {
                    "text": chunk.text,
                    "source": str(chunk.metadata.get("source_file") or path.name),
                    "metadata": dict(chunk.metadata),
                }
                for chunk in result.chunks
                if str(chunk.text or "").strip()
            )
        elif path.suffix.lower() == ".csv":
            records.extend(parse_csv(path))
        elif path.suffix.lower() in {".md", ".markdown"}:
            records.extend(parse_markdown(path))
        elif path.suffix.lower() == ".txt":
            records.extend(parse_text(path))
        else:
            logger.warning("Skipping unsupported file type: %s", path)
    return records


def _iter_supported_files(directory: Path) -> list[Path]:
    supported = {".pdf", ".csv", ".txt", ".md", ".markdown"}
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in supported)


def parse_pdf(pdf_path: Path, enrich_visuals: bool = True) -> list[dict[str, Any]]:
    text, figures = _extract_pdf_markdown_and_figures(pdf_path, extract_figures=enrich_visuals)
    chunks = _split_into_token_paragraphs(text, max_tokens=MAX_PARAGRAPH_TOKENS)
    records = [
        {
            "text": chunk,
            "source": pdf_path.name,
            "metadata": {
                "chunk_id": _stable_chunk_id(chunk),
                "document_type": "pdf",
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
                "chunk_index": index,
                "contains_table": False,
                "contains_chart": False,
            },
        }
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    ]
    if enrich_visuals and figures:
        records.extend(_caption_pdf_figures(pdf_path, figures, pdf_markdown_text=text))
    return attach_parent_context(enrich_records_with_cross_references(records))


def parse_text(text_path: Path) -> list[dict[str, Any]]:
    return _parse_plain_document(text_path, document_type="text")


def parse_markdown(markdown_path: Path) -> list[dict[str, Any]]:
    return _parse_plain_document(markdown_path, document_type="markdown")


def _parse_plain_document(text_path: Path, document_type: str) -> list[dict[str, Any]]:
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    chunks = _split_into_token_paragraphs(text, max_tokens=MAX_PARAGRAPH_TOKENS)
    return [
        {
            "text": chunk,
            "source": f"{text_path.name}#p{index}",
            "metadata": {
                "chunk_id": _stable_chunk_id(chunk),
                "document_type": document_type,
                "source_file": text_path.name,
                "source_path": str(text_path),
                "chunk_index": index,
                "contains_table": False,
                "contains_chart": False,
            },
        }
        for index, chunk in enumerate(chunks)
        if len(chunk.strip()) >= 10
    ]


def _extract_pdf_markdown_and_figures(pdf_path: Path, extract_figures: bool = True) -> tuple[str, list[dict[str, Any]]]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, OcrEngine, PdfPipelineOptions
        from docling.document_converter import DocumentConverter
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling_core.types.doc import PictureItem

        DOCLING_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(DOCLING_ARTIFACTS_PATH))

        pipeline_options = PdfPipelineOptions()
        pipeline_options.artifacts_path = DOCLING_ARTIFACTS_PATH
        pipeline_options.accelerator_options = AcceleratorOptions(device="cpu", num_threads=2)
        pipeline_options.generate_picture_images = extract_figures
        pipeline_options.generate_page_images = False
        pipeline_options.images_scale = 1.0
        # Keep Docling responsible for layout + figure isolation only.
        # Chart understanding is handled later by Gemini Vision so ingestion
        # stays stable on Windows and avoids local VLM dependencies.
        pipeline_options.do_chart_extraction = False
        pipeline_options.do_picture_classification = False
        pipeline_options.do_picture_description = False

        if DOCLING_OCR_ENGINE in {"off", "false", "0", "none", "disabled"}:
            pipeline_options.do_ocr = False
        elif DOCLING_OCR_ENGINE == OcrEngine.EASYOCR.value:
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=["en"],
                use_gpu=True,
                model_storage_directory=str(DOCLING_ARTIFACTS_PATH / "easyocr"),
            )

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        figure_paths = _save_docling_figures(result.document, pdf_path, PictureItem) if extract_figures else []
        return markdown, figure_paths
    except Exception as exc:
        logger.warning("Docling PDF extraction failed for %s; falling back to PyMuPDF: %s", pdf_path, exc)

    try:
        import fitz

        with fitz.open(str(pdf_path)) as document:
            return "\n\n".join(page.get_text("text") for page in document), []
    except Exception as exc:
        logger.warning("PyMuPDF PDF extraction failed for %s; falling back to pypdf: %s", pdf_path, exc)

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages), []
    except Exception as exc:
        raise RuntimeError(f"Could not extract PDF text from {pdf_path}: {exc}") from exc


def _save_docling_figures(document: object, pdf_path: Path, picture_type: type) -> list[dict[str, Any]]:
    EXTRACTED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []
    try:
        items = list(document.iterate_items())
    except Exception as exc:
        logger.warning("Could not iterate Docling document items for figures: %s", exc)
        return figures

    figure_index = 0
    for item, _level in items:
        if not isinstance(item, picture_type):
            continue
        figure_index += 1
        try:
            source_label = _docling_picture_label(item, document)
            entity_id = _entity_id_from_label(source_label, figure_index)
            image_path = EXTRACTED_CHARTS_DIR / f"{_sanitize_image_token(pdf_path.stem)}_{entity_id}.png"
            image = item.get_image(document)
            if image is None:
                continue
            image.save(image_path, "PNG")
            image_path_string = str(image_path)
            print(
                f"VALIDATION [Image Save]: {source_label} -> {image_path_string} "
                f"exists={Path(image_path_string).exists()}",
                flush=True,
            )
            figures.append(
                {
                    "image_path": image_path_string,
                    "source_label": source_label,
                    "entity_id": entity_id,
                    "figure_index": figure_index,
                }
            )
        except Exception as exc:
            logger.warning("Could not save PDF figure %s from %s: %s", figure_index, pdf_path, exc)
    return figures


def _docling_picture_label(item: object, document: object) -> str:
    """Fetch the human PDF label/caption attached to a Docling picture item."""

    try:
        caption_text = getattr(item, "caption_text", None)
        if callable(caption_text):
            caption = str(caption_text(document) or "").strip()
            if caption:
                return caption
    except Exception as exc:
        logger.debug("Docling picture caption_text lookup failed: %s", exc)

    for attr_name in ("caption", "caption_text", "text", "name", "label"):
        try:
            value = getattr(item, attr_name, None)
            if value and not callable(value):
                text = str(value).strip()
                if text:
                    return text
        except Exception:
            continue

    return "No explicit figure label or caption was found in the PDF layout metadata."


def _combine_source_label_and_visual_analysis(source_label: str, visual_analysis: str) -> str:
    clean_label = str(source_label or "").strip() or "No explicit figure label or caption was found in the PDF layout metadata."
    clean_analysis = str(visual_analysis or "").strip()
    return f"Source Label: {clean_label}\n\nVisual Analysis:\n{clean_analysis}"


def _find_surrounding_context(
    markdown_text: str,
    source_label: str,
    entity_id: str,
    context_window: int = 2,
) -> tuple[str, str]:
    """Find up to context_window paragraphs immediately before/after a figure reference."""
    if not markdown_text:
        return "", ""
    paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
    label_lower = source_label.lower() if source_label else ""
    entity_lower = entity_id.lower().replace("_", " ").replace("-", " ")

    match_index = -1
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        if label_lower and label_lower in para_lower:
            match_index = i
            break
        if entity_lower and entity_lower in para_lower:
            match_index = i
            break

    if match_index < 0:
        return "", ""

    before_paras = paragraphs[max(0, match_index - context_window): match_index]
    after_paras = paragraphs[match_index + 1: match_index + 1 + context_window]
    return "\n\n".join(before_paras), "\n\n".join(after_paras)


def _build_visual_chunk_text(
    *,
    source_label: str,
    entity_id: str,
    source_file: str,
    page: int | None,
    image_path: str,
    context_before: str,
    ocr_text: str,
    qwen_description: str,
    context_after: str,
) -> str:
    """Build the merged chunk text that combines all visual extraction outputs."""
    lines: list[str] = []
    header = source_label or entity_id or "Visual Element"
    lines.append(f"=== {header} ===")
    lines.append("")
    lines.append("[METADATA]")
    lines.append(f"  Source File : {source_file}")
    lines.append(f"  Entity ID   : {entity_id}")
    if page is not None:
        lines.append(f"  Page        : {page}")
    lines.append(f"  Image Path  : {image_path}")

    if context_before.strip():
        lines.append("")
        lines.append("[CONTEXT BEFORE]")
        lines.append(context_before.strip())

    lines.append("")
    lines.append("[PADDLE OCR -- Small Text Literals]")
    lines.append(ocr_text.strip() if ocr_text.strip() else "(no OCR text detected)")

    lines.append("")
    lines.append("[QWEN 2.5 VL -- Visual Analysis]")
    lines.append(qwen_description.strip() if qwen_description.strip() else "(no analysis generated)")

    if context_after.strip():
        lines.append("")
        lines.append("[CONTEXT AFTER]")
        lines.append(context_after.strip())

    return "\n".join(lines)


def _caption_pdf_figures(
    pdf_path: Path,
    figures: list[dict[str, Any]],
    pdf_markdown_text: str = "",
) -> list[dict[str, Any]]:
    """Caption each extracted PDF figure using PaddleOCR (CPU) + Qwen 2.5 VL (GPU).

    For each figure produces ONE merged chunk containing:
      - Metadata (source file, entity_id, page, image_path)
      - Context Before  (up to 2 PDF paragraphs above the figure reference)
      - PaddleOCR literals  (axis labels, tick values, legend items, exact numbers)
      - Qwen 2.5 VL analysis  (type-aware structured description)
      - Context After   (up to 2 PDF paragraphs below the figure reference)
    """
    if not figures:
        return []

    logger.info("Captioning %s extracted PDF visuals with PaddleOCR (CPU) + Qwen 2.5 VL (GPU)", len(figures))
    try:
        captioner = get_vision_captioner()
    except Exception as exc:
        logger.warning("Qwen Vision captioner unavailable; skipping all PDF visual captions: %s", exc)
        return []

    records: list[dict[str, Any]] = []
    for index, figure in enumerate(figures, start=1):
        image_path = Path(str(figure["image_path"]))
        image_path_string = str(image_path)
        entity_id = str(figure.get("entity_id") or _entity_id_from_label(figure.get("source_label", ""), index))
        source_label = str(figure.get("source_label") or "").strip()
        try:
            visual = ExtractedImage(
                image_path=image_path,
                page=None,
                type="chart",
                source_path=str(pdf_path),
                element_id=f"figure-{index}",
                metadata={
                    "source_file": pdf_path.name,
                    "figure_index": index,
                    "source_label": source_label,
                    "entity_id": entity_id,
                    "image_path": image_path_string,
                },
            )

            # --- surrounding paragraph context ---
            context_before, context_after = _find_surrounding_context(
                pdf_markdown_text, source_label, entity_id, context_window=1
            )

            # --- PaddleOCR + Qwen with context ---
            result = captioner.describe_image_with_context(
                visual,
                context_before=context_before,
                context_after=context_after,
            )
            if result is None:
                logger.warning("Skipping figure %s: captioner returned None", image_path)
                continue
            qwen_description, ocr_text = result

            if not qwen_description.strip() and not ocr_text.strip():
                logger.warning("Both Qwen and PaddleOCR returned empty output for %s", image_path)
                continue

            # --- assemble merged chunk ---
            chunk_text = _build_visual_chunk_text(
                source_label=source_label,
                entity_id=entity_id,
                source_file=pdf_path.name,
                page=None,
                image_path=image_path_string,
                context_before=context_before,
                ocr_text=ocr_text,
                qwen_description=qwen_description,
                context_after=context_after,
            )

            records.append(
                {
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "metadata": {
                        "chunk_id": _stable_chunk_id(f"{pdf_path.name}|{image_path.name}|{chunk_text}"),
                        "document_type": "pdf_visual",
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "image_path": image_path_string,
                        "image_local_path": image_path_string,
                        "entity_id": entity_id,
                        "figure_index": index,
                        "source_label": source_label,
                        "contains_table": False,
                        "contains_chart": True,
                        "vision_model": "Qwen2.5-VL-3B-AWQ",
                        "ocr_engine": "PaddleOCR",
                        "ocr_text": ocr_text,
                        "has_context_before": bool(context_before.strip()),
                        "has_context_after": bool(context_after.strip()),
                    },
                }
            )
        except Exception as exc:
            logger.warning(
                    "Vision captioning failed for PDF figure %s: %s",
                    image_path, exc,
                    exc_info=True,   # ← include full traceback in log
                )

    logger.info(
        "Generated %s merged visual chunks (PaddleOCR+Qwen) for %s",
        len(records),
        pdf_path,
    )
    return records


def _split_into_token_paragraphs(text: str, max_tokens: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in str(text or "").split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current).strip())
            current.clear()

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_tokens:
            flush()
            for start in range(0, len(words), max_tokens):
                chunks.append(" ".join(words[start : start + max_tokens]).strip())
            continue

        if len(current) + len(words) > max_tokens:
            flush()
        current.extend(words)
    flush()
    return chunks


def parse_csv(csv_path: Path) -> list[dict[str, Any]]:
    frame = _read_csv_with_detected_header(csv_path)
    frame = frame.dropna(axis=1, how="all")
    records: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        row_values = row.to_dict()
        row_text = _serialize_csv_row(csv_path.name, row_index, row_values)
        if not row_text:
            continue
        records.append(
            {
                "text": row_text,
                "source": csv_path.name,
                "metadata": {
                    "chunk_id": _stable_chunk_id(row_text),
                    "document_type": "csv",
                    "source_file": csv_path.name,
                    "source_path": str(csv_path),
                    "row_id": int(row_index),
                    "columns": list(frame.columns),
                    "contains_table": True,
                    "contains_chart": False,
                },
            }
        )
    return records


def _read_csv_with_detected_header(csv_path: Path) -> pd.DataFrame:
    """Load CSVs that may contain export metadata before the actual table header."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    for row_index, row in enumerate(rows[:50]):
        normalized = {cell.strip().lower() for cell in row}
        if {"country name", "country code"}.issubset(normalized):
            return pd.read_csv(csv_path, skiprows=row_index)

    return pd.read_csv(csv_path)


def _serialize_csv_row(file_name: str, row_index: int, row_values: dict[str, Any]) -> str:
    metadata_facts: list[str] = []
    metric_facts: list[str] = []

    for column, value in row_values.items():
        if _is_missing_value(column) or _is_missing_value(value):
            continue

        column_text = str(column).strip()
        value_text = str(value).strip()
        if column_text.isdigit():
            metric_facts.append(f"In the year {column_text}, the value was {value_text}.")
        else:
            metadata_facts.append(f"{column_text}: {value_text}")

    if not metadata_facts and not metric_facts:
        return ""

    summary_value = _csv_summary_value(row_values)
    metadata_text = ", ".join(metadata_facts)
    metric_text = " ".join(metric_facts)
    return (
        f"Data Sheet Metric Lookup -> File: {file_name}, Row ID: {row_index}, "
        f"{metadata_text}. {metric_text} Context/Trend Summary: {summary_value}"
    )


def _csv_summary_value(row_values: dict[str, Any]) -> str:
    lowered = {
        str(key).lower(): value
        for key, value in row_values.items()
        if not _is_missing_value(key) and not _is_missing_value(value)
    }
    country = lowered.get("country") or lowered.get("country name") or lowered.get("region")
    year = lowered.get("year") or lowered.get("date")
    gdp = lowered.get("gdp") or lowered.get("gdp value") or lowered.get("revenue")
    if country not in (None, "") and year not in (None, "") and gdp not in (None, ""):
        return f"{country} had value {gdp} in {year}."
    return next((str(value) for value in row_values.values() if not _is_missing_value(value)), "structured row data")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == "" or str(value).lower().startswith("unnamed:")


def encode_records(
    model: BgeM3EmbeddingModel,
    records: list[dict[str, Any]],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_length: int = EMBEDDING_MAX_LENGTH,
) -> dict[str, Any]:
    texts = [record["text"] for record in records]
    logger.info("Encoding %s records with pure Transformers BGE-M3 dense output", len(texts))
    dense_vectors = model.embed_documents(texts, batch_size=batch_size)
    return {
        "dense_vecs": dense_vectors,
        "lexical_weights": [{} for _ in dense_vectors],
    }


def _bge_sparse_to_qdrant(sparse_weights: dict[Any, Any]) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(index) for index in sparse_weights.keys()],
        values=[float(value) for value in sparse_weights.values()],
    )


def build_points(records: list[dict[str, Any]], embeddings: dict[str, Any]) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    dense_vectors = embeddings.get("dense_vecs")
    if dense_vectors is None:
        dense_vectors = embeddings.get("dense")
    if dense_vectors is None:
        raise ValueError("BGE-M3 output did not include dense embeddings.")
    sparse_vectors = embeddings["lexical_weights"]

    for index, record in enumerate(records):
        text = str(record["text"])
        source = str(record["source"])
        dense = [float(value) for value in dense_vectors[index]]
        if len(dense) != DENSE_VECTOR_SIZE:
            raise ValueError(f"Expected dense vector dim {DENSE_VECTOR_SIZE}, got {len(dense)}")

        metadata = enrich_chunk_metadata(dict(record.get("metadata") or {}), text)
        metadata = _normalize_visual_metadata_paths(metadata)
        metadata.setdefault("chunk_id", _stable_chunk_id(text))
        metadata.setdefault("source", source)
        metadata.setdefault("contains_table", metadata.get("document_type") == "csv")
        metadata.setdefault("contains_chart", False)
        if _requires_visual_image_path(metadata):
            _verified_chart_image_path(metadata)
            print(
                f"VALIDATION [Qdrant Payload]: chunk_id={metadata.get('chunk_id')} "
                f"contains_chart={metadata.get('contains_chart')} contains_table={metadata.get('contains_table')} "
                f"image_path={metadata.get('image_path')} exists=True",
                flush=True,
            )
        payload = {
            "text": text,
            "page_content": text,
            "source": source,
            "contains_chart": bool(metadata.get("contains_chart")),
            "contains_table": bool(metadata.get("contains_table")),
            "contains_figure": bool(metadata.get("contains_figure")),
            "contains_image": bool(metadata.get("contains_image")),
            "contains_csv": bool(metadata.get("contains_csv")),
            "metadata": metadata,
        }
        if metadata.get("image_path") not in ("", None):
            payload["image_path"] = metadata["image_path"]
        for key in ASSET_FIELDS:
            if metadata.get(key) not in ("", None, [], {}):
                payload[key] = metadata[key]
        if not str(payload.get("text") or "").strip():
            raise ValueError(f"Cannot upsert record without root payload['text']; source={source}")
        points.append(
            models.PointStruct(
                id=metadata["chunk_id"],
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: _bge_sparse_to_qdrant(sparse_vectors[index]),
                },
                payload=payload,
            )
        )
    return points


def _stable_chunk_id(chunk_content: str) -> str:
    """Create deterministic point IDs from raw chunk text for idempotent upserts."""

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_content)))


def upsert_points(client: QdrantClient, points: list[models.PointStruct]) -> int:
    uploaded = 0
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[start : start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded += len(batch)
        logger.info("Uploaded %s/%s points", uploaded, len(points))
    return uploaded


def _record_batches(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def ingest_data(
    paths: Iterable[str | Path],
    recreate_collection: bool = False,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
    embedding_max_length: int = EMBEDDING_MAX_LENGTH,
    enrich_pdf_visuals: bool = True,
) -> int:
    client = get_qdrant_client()
    create_or_recreate_collection(client, recreate=recreate_collection)
    records = parse_sources(paths, enrich_pdf_visuals=enrich_pdf_visuals)
    if not records:
        logger.warning("No records parsed for ingestion.")
        return 0

    visual_caption_count = sum(
        1
        for record in records
        if (record.get("metadata") or {}).get("document_type") == "pdf_visual"
    )
    print(f"Total chunks parsed: {len(records)}")
    print(f"Visual captions generated: {visual_caption_count}")

    model = get_embedding_model()
    uploaded = 0
    total_records = len(records)
    for batch_index, record_batch in enumerate(_record_batches(records, embedding_batch_size), start=1):
        logger.info(
            "Embedding and upserting record batch %s containing %s records",
            batch_index,
            len(record_batch),
        )
        embeddings = encode_records(
            model,
            record_batch,
            batch_size=embedding_batch_size,
            max_length=embedding_max_length,
        )
        points = build_points(record_batch, embeddings)
        uploaded += upsert_points(client, points)
        logger.info("Total uploaded after batch %s: %s/%s", batch_index, uploaded, total_records)

    print_collection_point_count(client)
    return uploaded


def print_collection_point_count(client: QdrantClient | None = None) -> int | None:
    """Print Qdrant's persisted point count for quick ingestion verification."""

    try:
        active_client = client or get_qdrant_client()
        collection_info = active_client.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        print(f"Qdrant collection '{COLLECTION_NAME}' points_count: {points_count}")
        return points_count
    except Exception as exc:
        logger.error("Could not read Qdrant collection point count: %s", exc)
        return None


def process_and_upload_datasets(data_directory: str = "./Data", recreate_collection: bool = True) -> int:
    """Reset the local named-vector collection and ingest all supported files in a directory."""

    return ingest_data([data_directory], recreate_collection=recreate_collection)


def main() -> None:
    global MAX_PARAGRAPH_TOKENS

    parser = argparse.ArgumentParser(description="Ingest PDFs, CSVs, and TXT files into local Qdrant with BGE-M3 dense+sparse vectors.")
    parser.add_argument("sources", nargs="*", default=["./Data"], help="Files or directories to ingest.")
    parser.add_argument("--preserve", action="store_true", help="Preserve the existing collection instead of recreating it.")
    parser.add_argument("--count-only", action="store_true", help="Print the current Qdrant point count without ingesting data.")
    parser.add_argument("--chunk-tokens", type=int, default=MAX_PARAGRAPH_TOKENS, help="Target text/Markdown/PDF chunk size; use 512 to 1024 for RAG indexing.")
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE, help="BGE-M3 embedding batch size.")
    parser.add_argument("--embedding-max-length", type=int, default=EMBEDDING_MAX_LENGTH, help="BGE-M3 max sequence length; keep 512 to 1024 for this pipeline.")
    parser.add_argument("--skip-pdf-visuals", action="store_true", help="Skip Gemini Vision captions for figures embedded inside PDFs.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    if not 512 <= args.chunk_tokens <= 1024:
        raise ValueError("--chunk-tokens must be between 512 and 1024.")
    if not 512 <= args.embedding_max_length <= 1024:
        raise ValueError("--embedding-max-length must be between 512 and 1024.")
    MAX_PARAGRAPH_TOKENS = args.chunk_tokens

    if args.count_only:
        print_collection_point_count()
        return

    count = ingest_data(
        args.sources,
        recreate_collection=not args.preserve,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        enrich_pdf_visuals=not args.skip_pdf_visuals,
    )
    print(f"Ingestion completed. Upserted {count} points into {COLLECTION_NAME}.")


if __name__ == "__main__":
    main()
