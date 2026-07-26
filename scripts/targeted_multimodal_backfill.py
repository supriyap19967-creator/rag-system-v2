from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
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

from qdrant_client import QdrantClient

from ingest_data import COLLECTION_NAME, DOCLING_ARTIFACTS_PATH, DOCLING_OCR_ENGINE

DEFAULT_PDF_PATH = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "assets" / "extracted_images"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "assets" / "extracted_tables"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
PAGE_WINDOW_SIZE = 2

ENTITY_PATTERN = re.compile(
    r"\b(?P<kind>Figure|Fig\.?|Chart|Diagram|Table)\s*(?P<number>[A-Za-z]?\d+(?:\.\d+)*)\b",
    flags=re.IGNORECASE,
)
VISUAL_WORD_PATTERN = re.compile(r"\b(figure|fig\.?|chart|diagram|image)\b", flags=re.IGNORECASE)
TABLE_WORD_PATTERN = re.compile(r"\btable\b", flags=re.IGNORECASE)

logger = logging.getLogger(__name__)


@dataclass
class PointGap:
    point_id: Any
    payload: dict[str, Any]
    text: str
    metadata: dict[str, Any]
    page_no: int
    entity_id: str
    needs_image: bool = False
    needs_table: bool = False


@dataclass
class PageAssets:
    page_no: int
    images: dict[str, str] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    visual_headers: dict[str, str] = field(default_factory=dict)
    fallback_images: list[str] = field(default_factory=list)
    fallback_tables: list[str] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted in-place Qdrant multimodal asset backfill.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="PDF to use for targeted page extraction")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR))
    parser.add_argument("--limit-pages", type=int, default=0, help="Optional maximum page count to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Scan and extract but do not update Qdrant")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")

    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract is not visible to this Python process. "
            "Set TESSERACT_DIR or add C:\\Program Files\\Tesseract-OCR to PATH."
        )

    pdf_path = Path(args.pdf).resolve()
    image_dir = Path(args.image_dir).resolve()
    table_dir = Path(args.table_dir).resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(url=args.qdrant_url)
    gaps_by_page = find_gap_points(client, args.collection)
    if args.limit_pages:
        keep_pages = sorted(gaps_by_page)[: args.limit_pages]
        gaps_by_page = {page_no: gaps_by_page[page_no] for page_no in keep_pages}

    total_points = sum(len(points) for points in gaps_by_page.values())
    logger.info("Identified %s pages and %s Qdrant points needing backfill.", len(gaps_by_page), total_points)
    if not gaps_by_page:
        return

    updated_points = 0
    for start_page, end_page in progress_iter(_page_windows(sorted(gaps_by_page), PAGE_WINDOW_SIZE), len(gaps_by_page)):
        target_pages = [page_no for page_no in range(start_page, end_page + 1) if page_no in gaps_by_page]
        logger.info("Backfilling page window %s-%s; target_pages=%s", start_page, end_page, target_pages)
        assets_by_page = extract_window_assets(
            pdf_path=pdf_path,
            page_range=(start_page, end_page),
            target_pages=target_pages,
            page_gaps=gaps_by_page,
            image_dir=image_dir,
            table_dir=table_dir,
        )
        for page_no in target_pages:
            updated_points += update_page_points(
                client=client,
                collection_name=args.collection,
                gaps=gaps_by_page[page_no],
                assets=assets_by_page.get(page_no, PageAssets(page_no=page_no)),
                dry_run=args.dry_run,
            )

    logger.info("Targeted backfill finished. Updated %s existing Qdrant points.", updated_points)


def find_gap_points(client: QdrantClient, collection_name: str) -> dict[int, list[PointGap]]:
    gaps_by_page: dict[int, list[PointGap]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata") or {})
            text = _point_text(payload)
            if not text:
                continue
            entity = _extract_entity(metadata.get("entity_id") or text)
            references_visual = bool(VISUAL_WORD_PATTERN.search(text))
            references_table = bool(TABLE_WORD_PATTERN.search(text))
            if not references_visual and not references_table:
                continue
            page_no = _page_no(metadata)
            if page_no is None:
                continue

            entity_id = str(metadata.get("entity_id") or (entity[0] if entity else f"Page_{page_no}_asset"))
            needs_image = references_visual and not _valid_existing_path(
                metadata.get("image_path") or metadata.get("image_local_path") or payload.get("image_path")
            )
            needs_table = references_table and not _valid_existing_path(
                metadata.get("table_csv_path")
                or metadata.get("csv_path")
                or payload.get("table_csv_path")
                or payload.get("csv_path")
            )
            if not needs_image and not needs_table:
                continue

            gaps_by_page.setdefault(page_no, []).append(
                PointGap(
                    point_id=point.id,
                    payload=payload,
                    text=text,
                    metadata=metadata,
                    page_no=page_no,
                    entity_id=entity_id,
                    needs_image=needs_image,
                    needs_table=needs_table,
                )
            )
        if offset is None:
            break
    return gaps_by_page


def extract_window_assets(
    *,
    pdf_path: Path,
    page_range: tuple[int, int],
    target_pages: list[int],
    page_gaps: dict[int, list[PointGap]],
    image_dir: Path,
    table_dir: Path,
) -> dict[int, PageAssets]:
    assets_by_page = {page_no: PageAssets(page_no=page_no) for page_no in target_pages}
    document = None
    try:
        document = convert_pdf_with_docling(pdf_path, page_range=page_range)
        collect_docling_assets(document, pdf_path, assets_by_page, image_dir, table_dir)
    except Exception as exc:
        logger.warning("Docling targeted conversion failed for pages %s-%s: %s", page_range[0], page_range[1], exc)

    for page_no in target_pages:
        missing_image_entities = [
            gap.entity_id for gap in page_gaps.get(page_no, []) if gap.needs_image and gap.entity_id not in assets_by_page[page_no].images
        ]
        if missing_image_entities:
            collect_unstructured_crops(pdf_path, page_no, missing_image_entities, assets_by_page[page_no], image_dir)
    return assets_by_page


def convert_pdf_with_docling(pdf_path: Path, page_range: tuple[int, int]) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, OcrEngine, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    DOCLING_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(DOCLING_ARTIFACTS_PATH))

    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = DOCLING_ARTIFACTS_PATH
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = False
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.do_table_structure = True
    pipeline_options.do_chart_extraction = False
    pipeline_options.do_picture_classification = False
    pipeline_options.do_picture_description = False

    if DOCLING_OCR_ENGINE in {"off", "false", "0", "none", "disabled"}:
        pipeline_options.do_ocr = False
    elif DOCLING_OCR_ENGINE == OcrEngine.EASYOCR.value:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=False,
            model_storage_directory=str(DOCLING_ARTIFACTS_PATH / "easyocr"),
        )

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    return converter.convert(str(pdf_path), page_range=page_range).document


def collect_docling_assets(
    document: Any,
    pdf_path: Path,
    assets_by_page: dict[int, PageAssets],
    image_dir: Path,
    table_dir: Path,
) -> None:
    page_text_by_entity: dict[tuple[int, str], list[str]] = {}
    for item, _level in document.iterate_items():
        page_no = _item_page_no(item)
        if page_no not in assets_by_page:
            continue
        text = _item_text(item, document)
        entity = _extract_entity(text)
        if entity:
            entity_id, kind = entity
            page_text_by_entity.setdefault((page_no, entity_id), []).append(text)
            if kind == "table":
                csv_path = save_table_csv(table_dir, page_no, entity_id, "\n\n".join(page_text_by_entity[(page_no, entity_id)]))
                assets_by_page[page_no].tables[entity_id] = csv_path
                assets_by_page[page_no].fallback_tables.append(csv_path)
            else:
                assets_by_page[page_no].visual_headers.setdefault(entity_id, _label_from_entity_id(entity_id))
        if _is_docling_picture_item(item):
            entity_id = entity[0] if entity and entity[1] == "figure" else f"Figure_page_{page_no}"
            image_path = save_docling_picture(item, document, image_dir, page_no, entity_id)
            if image_path:
                assets_by_page[page_no].images[entity_id] = image_path
                assets_by_page[page_no].visual_headers.setdefault(entity_id, _header_from_text(text, entity_id))
                assets_by_page[page_no].fallback_images.append(image_path)


def collect_unstructured_crops(
    pdf_path: Path,
    page_no: int,
    entity_ids: list[str],
    assets: PageAssets,
    image_dir: Path,
) -> None:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        logger.warning("unstructured[pdf] is unavailable; cannot backfill page %s: %s", page_no, exc)
        apply_full_page_fallback(pdf_path, page_no, entity_ids, assets, image_dir)
        return

    tmp_dir = image_dir / "_backfill_tmp" / f"page_{page_no}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    page_pdf = _single_page_pdf(pdf_path, page_no, tmp_dir)

    try:
        elements = partition_pdf(
            filename=str(page_pdf),
            strategy=os.getenv("PDF_VISUAL_STRATEGY", "hi_res"),
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=str(tmp_dir),
        )
    except Exception as exc:
        logger.warning("Unstructured crop extraction failed for page %s: %s", page_no, exc)
        apply_full_page_fallback(pdf_path, page_no, entity_ids, assets, image_dir)
        return

    source_paths = _extracted_image_paths(elements)
    missing_entities: list[str] = []
    for index, entity_id in enumerate(entity_ids):
        if entity_id in assets.images:
            continue
        source_path = source_paths[index] if index < len(source_paths) else None
        if not source_path or not source_path.exists():
            logger.warning("No extracted crop available for page=%s entity=%s", page_no, entity_id)
            missing_entities.append(entity_id)
            continue
        final_path = image_dir / f"page_{page_no}_{_path_token(entity_id)}.png"
        shutil.copy2(source_path, final_path)
        resolved = str(final_path.resolve())
        assets.images[entity_id] = resolved
        assets.visual_headers.setdefault(entity_id, _label_from_entity_id(entity_id))
        assets.fallback_images.append(resolved)
        logger.info("Saved backfilled visual crop: entity=%s path=%s", entity_id, resolved)

    if missing_entities:
        apply_full_page_fallback(pdf_path, page_no, missing_entities, assets, image_dir)


def apply_full_page_fallback(
    pdf_path: Path,
    page_no: int,
    entity_ids: list[str],
    assets: PageAssets,
    image_dir: Path,
) -> None:
    pending = [entity_id for entity_id in entity_ids if entity_id not in assets.images]
    if not pending:
        return
    fallback_path, page_headers = render_layout_content_fallback(pdf_path, page_no, image_dir)
    if not fallback_path:
        return
    for entity_id in pending:
        assets.images[entity_id] = fallback_path
        assets.visual_headers[entity_id] = _header_for_entity(entity_id, page_headers)
    if fallback_path not in assets.fallback_images:
        assets.fallback_images.append(fallback_path)
    message = f"Page {page_no}: No structural crop found. Successfully applied layout-aware content crop fallback."
    logger.info(message)
    print(message, flush=True)


def render_layout_content_fallback(
    pdf_path: Path,
    page_no: int,
    image_dir: Path,
    scale: float = 3.0,
) -> tuple[str, list[str]]:
    try:
        import fitz
    except ImportError as exc:
        logger.warning("PyMuPDF is unavailable; cannot render content crop fallback for page %s: %s", page_no, exc)
        return "", []

    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / f"page_{page_no}_full_page_fallback.png"
    document = None
    try:
        document = fitz.open(str(pdf_path))
        page = document.load_page(page_no - 1)
        text_blocks = page.get_text("blocks") or []
        drawings = page.get_drawings() or []
        bbox = _content_bbox_from_layout(page.rect, text_blocks, drawings)
        headers = _semantic_headers_from_blocks(page.rect, text_blocks)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=bbox, alpha=False)
        pixmap.save(str(output_path))
    except Exception as exc:
        logger.warning("Could not render content crop fallback for page %s: %s", page_no, exc)
        return "", []
    finally:
        if document is not None:
            document.close()
    return str(output_path.resolve()), headers


def _single_page_pdf(pdf_path: Path, page_no: int, output_dir: Path) -> Path:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_no - 1])
    target = output_dir / f"source_page_{page_no}.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    return target


def update_page_points(
    *,
    client: QdrantClient,
    collection_name: str,
    gaps: list[PointGap],
    assets: PageAssets,
    dry_run: bool,
) -> int:
    updated = 0
    for gap in gaps:
        metadata = dict(gap.metadata)
        payload_update = dict(gap.payload)
        changed = False

        if gap.needs_image:
            image_path = assets.images.get(gap.entity_id) or _first(assets.fallback_images)
            if image_path:
                visual_anchor_header = (
                    assets.visual_headers.get(gap.entity_id)
                    or _header_from_text(gap.text, gap.entity_id)
                    or _label_from_entity_id(gap.entity_id)
                )
                metadata["image_path"] = image_path
                metadata["image_local_path"] = image_path
                metadata["final_image_path"] = image_path
                metadata["contains_chart"] = True
                metadata["visual_anchor_header"] = visual_anchor_header
                metadata["visual_binding"] = {
                    "image_path": image_path,
                    "visual_anchor_header": visual_anchor_header,
                }
                payload_update["image_path"] = image_path
                payload_update["contains_chart"] = True
                payload_update["visual_anchor_header"] = visual_anchor_header
                payload_update["visual_binding"] = metadata["visual_binding"]
                changed = True

        if gap.needs_table:
            csv_path = assets.tables.get(gap.entity_id) or _first(assets.fallback_tables)
            if csv_path:
                metadata["csv_path"] = csv_path
                metadata["table_csv_path"] = csv_path
                metadata["contains_table"] = True
                payload_update["csv_path"] = csv_path
                payload_update["table_csv_path"] = csv_path
                payload_update["contains_table"] = True
                changed = True

        if not changed:
            logger.warning("No asset found for point=%s page=%s entity=%s", gap.point_id, gap.page_no, gap.entity_id)
            continue

        payload_update["metadata"] = metadata
        if dry_run:
            logger.info("[dry-run] Would update point=%s payload_keys=%s", gap.point_id, sorted(payload_update))
        else:
            client.set_payload(collection_name=collection_name, payload=payload_update, points=[gap.point_id], wait=True)
            logger.info("Updated Qdrant point=%s page=%s entity=%s", gap.point_id, gap.page_no, gap.entity_id)
        updated += 1
    return updated


def save_docling_picture(item: Any, document: Any, image_dir: Path, page_no: int, entity_id: str) -> str:
    image_getter = getattr(item, "get_image", None)
    if not callable(image_getter):
        return ""
    try:
        image = image_getter(document)
    except Exception as exc:
        logger.warning("Docling get_image failed for page=%s entity=%s: %s", page_no, entity_id, exc)
        return ""
    if image is None:
        return ""
    image_path = image_dir / f"page_{page_no}_{_path_token(entity_id)}.png"
    try:
        image.save(image_path, "PNG")
    except Exception as exc:
        logger.warning("Could not save Docling image page=%s entity=%s: %s", page_no, entity_id, exc)
        return ""
    resolved = str(image_path.resolve())
    logger.info("Saved backfilled Docling image: entity=%s path=%s", entity_id, resolved)
    return resolved


def save_table_csv(table_dir: Path, page_no: int, entity_id: str, text: str) -> str:
    rows = _markdown_table_rows(text)
    if not rows:
        rows = [["entity_id", "page_no", "context"], [entity_id, str(page_no), text.strip()]]
    csv_path = table_dir / f"page_{page_no}_{_path_token(entity_id)}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    resolved = str(csv_path.resolve())
    logger.info("Saved backfilled table CSV: entity=%s path=%s", entity_id, resolved)
    return resolved


def progress_iter(windows: Iterable[tuple[int, int]], total_pages: int) -> Iterable[tuple[int, int]]:
    try:
        from tqdm import tqdm

        completed_pages = 0
        for start_page, end_page in tqdm(list(windows), desc="Backfilling pages", unit="window"):
            yield start_page, end_page
            completed_pages += end_page - start_page + 1
            tqdm.write(f"Backfill progress: {min(completed_pages, total_pages)}/{total_pages} pages")
    except ImportError:
        for start_page, end_page in windows:
            print(f"Backfilling pages {start_page}-{end_page}", flush=True)
            yield start_page, end_page


def _page_windows(page_numbers: list[int], window_size: int) -> Iterable[tuple[int, int]]:
    if not page_numbers:
        return
    group_start = page_numbers[0]
    previous = page_numbers[0]
    group = [page_numbers[0]]
    for page_no in page_numbers[1:]:
        if page_no == previous + 1 and len(group) < window_size:
            group.append(page_no)
        else:
            yield group_start, previous
            group_start = page_no
            group = [page_no]
        previous = page_no
    yield group_start, previous


def _point_text(payload: dict[str, Any]) -> str:
    return str(payload.get("text") or payload.get("page_content") or payload.get("content") or "")


def _page_no(metadata: dict[str, Any]) -> int | None:
    for key in ("page_no", "page", "docling_page_no"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_entity(text: Any) -> tuple[str, str] | None:
    match = ENTITY_PATTERN.search(str(text or ""))
    if not match:
        return None
    kind_raw = match.group("kind").lower().rstrip(".")
    number = match.group("number")
    kind = "table" if kind_raw == "table" else "figure"
    label = "Table" if kind == "table" else "Figure"
    return f"{label}_{number}", kind


def _content_bbox_from_layout(page_rect: Any, text_blocks: list[Any], drawings: list[Any]) -> Any:
    import fitz

    valid_rects: list[Any] = []
    for block in text_blocks:
        if len(block) < 5:
            continue
        rect = fitz.Rect(block[:4])
        text = str(block[4] or "").strip()
        if _is_noise_zone(rect, text, page_rect):
            continue
        if rect.width >= 4 and rect.height >= 4:
            valid_rects.append(rect)

    for drawing in drawings:
        raw_rect = drawing.get("rect") if isinstance(drawing, dict) else None
        if raw_rect is None:
            continue
        rect = fitz.Rect(raw_rect)
        if rect.is_empty or rect.width < 6 or rect.height < 6:
            continue
        if _is_noise_zone(rect, "", page_rect):
            continue
        valid_rects.append(rect)

    if not valid_rects:
        return page_rect

    bbox = fitz.Rect(valid_rects[0])
    for rect in valid_rects[1:]:
        bbox.include_rect(rect)

    padding = 12
    bbox = fitz.Rect(bbox.x0 - padding, bbox.y0 - padding, bbox.x1 + padding, bbox.y1 + padding)
    return bbox & page_rect


def _is_noise_zone(rect: Any, text: str, page_rect: Any) -> bool:
    top_limit = page_rect.height * 0.10
    bottom_limit = page_rect.height * 0.90
    stripped = re.sub(r"\s+", " ", str(text or "").strip())
    lower = stripped.lower()
    looks_repetitive = (
        not stripped
        or bool(re.fullmatch(r"\d+|[ivxlcdm]+", stripped, flags=re.IGNORECASE))
        or "world development report" in lower
        or "chapter" in lower and len(stripped) < 70
    )
    if rect.y1 <= top_limit and looks_repetitive:
        return True
    if rect.y0 >= bottom_limit and looks_repetitive:
        return True
    return False


def _semantic_headers_from_blocks(page_rect: Any, text_blocks: list[Any]) -> list[str]:
    headers: list[str] = []
    for block in sorted(text_blocks, key=lambda item: (item[1], item[0]) if len(item) >= 2 else (0, 0)):
        if len(block) < 5:
            continue
        text = re.sub(r"\s+", " ", str(block[4] or "").strip())
        if not text:
            continue
        rect = type(page_rect)(block[:4])
        if _is_noise_zone(rect, text, page_rect):
            continue

        entity_header = ENTITY_PATTERN.search(text)
        if entity_header:
            headers.append(_normalise_header(text))
            continue

        line_count = max(1, str(block[4] or "").count("\n") + 1)
        is_short_heading = len(text) <= 120 and line_count <= 3
        has_heading_shape = (
            bool(re.match(r"^(\d+(?:\.\d+)*|[A-Z][A-Za-z]+)\s+[\w,( -]+$", text))
            or text.istitle()
            or text.isupper()
        )
        if is_short_heading and has_heading_shape and not text.endswith("."):
            headers.append(_normalise_header(text))
    return _dedupe(headers)


def _header_for_entity(entity_id: str, headers: list[str]) -> str:
    label = _label_from_entity_id(entity_id)
    label_pattern = re.compile(re.escape(label).replace(r"\ ", r"\s+"), flags=re.IGNORECASE)
    compact_label = label.replace(" ", "")
    for header in headers:
        if label_pattern.search(header) or compact_label.lower() in header.replace(" ", "").lower():
            return header
    return headers[0] if headers else label


def _header_from_text(text: Any, entity_id: str) -> str:
    source = str(text or "")
    label = _label_from_entity_id(entity_id)
    entity_match = ENTITY_PATTERN.search(source)
    if entity_match:
        start = max(0, source.rfind("\n", 0, entity_match.start()) + 1)
        end = source.find("\n", entity_match.end())
        if end == -1:
            end = min(len(source), entity_match.end() + 160)
        return _normalise_header(source[start:end]) or label
    return label


def _label_from_entity_id(entity_id: str) -> str:
    value = str(entity_id or "").strip()
    match = re.match(r"^(Figure|Table|Chart|Diagram)[_\s-]*(.+)$", value, flags=re.IGNORECASE)
    if not match:
        return value or "Visual asset"
    kind = match.group(1).title()
    number = match.group(2).replace("_", ".").replace("-", ".").strip(". ")
    if kind in {"Chart", "Diagram"}:
        kind = "Figure"
    return f"{kind} {number}".strip()


def _normalise_header(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:180]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _valid_existing_path(value: Any) -> bool:
    if not value:
        return False
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, DEFAULT_IMAGE_DIR / path.name, DEFAULT_TABLE_DIR / path.name])
    return any(candidate.exists() for candidate in candidates)


def _item_page_no(item: Any) -> int | None:
    prov = list(getattr(item, "prov", []) or [])
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return int(page_no) if page_no is not None else None


def _item_text(item: Any, document: Any) -> str:
    if hasattr(item, "text") and getattr(item, "text"):
        return str(getattr(item, "text")).strip()
    export_to_markdown = getattr(item, "export_to_markdown", None)
    if callable(export_to_markdown):
        try:
            return str(export_to_markdown(document) or "").strip()
        except Exception:
            pass
    caption_text = getattr(item, "caption_text", None)
    if callable(caption_text):
        try:
            return str(caption_text(document) or "").strip()
        except Exception:
            return ""
    return ""


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
    rows: list[list[str]] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and not all(set(cell) <= {"-", ":"} for cell in cells):
            rows.append(cells)
    return rows


def _path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", str(value or "")).strip("_") or "asset"


def _first(values: list[str]) -> str:
    return values[0] if values else ""


if __name__ == "__main__":
    main()
