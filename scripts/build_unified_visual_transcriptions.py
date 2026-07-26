from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz

from ingestion.config import IngestionSettings
from ingestion.paddle_ocr import PaddleOcrExtractor
from ingestion.qwen_vision_caption import QwenVisionCaptioner
from ingestion.schemas import ExtractedImage


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_DIR = ROOT / "Data" / "Pdf"
DEFAULT_IMAGE_DIR = ROOT / "assets" / "extracted_images"
DEFAULT_OUTPUT_DIR = ROOT / "data_cache" / "transcriptions"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

ENTITY_PATTERN = re.compile(
    r"\b(?P<kind>figure|fig\.?|table|chart|diagram|image|map)\s*[_\-\s]?(?P<identifier>[A-Za-z]?\d+(?:\.\d+)*(?:[A-Za-z]?)?)\b",
    flags=re.IGNORECASE,
)
PAGE_PATTERN = re.compile(r"\bpage[_\-\s]?(?P<page>\d+)\b", flags=re.IGNORECASE)
BOILERPLATE_PATTERN = re.compile(
    r"\b(?:http|www\.|reproducibility|replication|github|bibliography|references|copyright|doi|isbn|issn)\b",
    re.IGNORECASE,
)
SOURCE_NOTE_PATTERN = re.compile(r"^(Source|Sources|Note|Notes)\s*:\s*(?P<body>.+)$", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class UnifiedVisualTranscription:
    chunk_id: str
    page_content: str
    metadata: dict[str, object]
    nearby_context_paragraphs: str
    raw_ocr_literals: str
    qwen_visual_analysis: str


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_entity_id(kind: str, identifier: str) -> str:
    prefix = kind.title()
    if prefix == "Fig":
        prefix = "Figure"
    if prefix == "Table":
        prefix = "Table"
    if prefix == "Chart":
        prefix = "Chart"
    if prefix == "Diagram":
        prefix = "Diagram"
    if prefix == "Image":
        prefix = "Image"
    if prefix == "Map":
        prefix = "Map"
    normalized_identifier = re.sub(r"\s+", "", str(identifier or "")).strip()
    return f"{prefix}_{normalized_identifier}" if normalized_identifier else prefix


def _infer_entity_id_from_name(path: Path) -> str:
    match = ENTITY_PATTERN.search(path.stem.replace("_", " "))
    if match:
        return _normalize_entity_id(match.group("kind"), match.group("identifier"))
    page_match = PAGE_PATTERN.search(path.stem)
    page_no = page_match.group("page") if page_match else "unknown"
    return f"Figure_page_{page_no}"


def _infer_visual_type(path: Path, entity_id: str) -> str:
    stem = path.stem.lower()
    if "table" in stem or entity_id.lower().startswith("table_"):
        return "table"
    if "chart" in stem or entity_id.lower().startswith("chart_"):
        return "chart"
    if "diagram" in stem or entity_id.lower().startswith("diagram_"):
        return "diagram"
    if "map" in stem or entity_id.lower().startswith("map_"):
        return "map"
    return "figure"


def _page_no_from_path(path: Path) -> int | None:
    match = PAGE_PATTERN.search(path.stem)
    if not match:
        return None
    try:
        return int(match.group("page"))
    except ValueError:
        return None


def _is_processable_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
        and ".raw" not in path.name.lower()
        and "full_page_fallback" not in path.name.lower()
    )


def _literal_csv_text(text: str) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for line in (segment.strip() for segment in str(text or "").splitlines()):
        if not line:
            continue
        normalized = _clean_text(line)
        if normalized and normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    return ", ".join(items)


def _extract_pdf_blocks(page: fitz.Page) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for block in sorted(page.get_text("blocks"), key=lambda item: (item[1], item[0])):
        x0, y0, x1, y1, text, *_rest = block
        clean = _clean_text(text)
        if not clean:
            continue
        blocks.append(
            {
                "text": clean,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
            }
        )
    return blocks


def _find_anchor_index(blocks: list[dict[str, object]], entity_id: str, visual_type: str) -> int:
    normalized_entity = re.sub(r"[^a-z0-9.]+", " ", entity_id.lower()).strip()
    best_index = -1
    best_score = -1
    for index, block in enumerate(blocks):
        text = str(block.get("text") or "")
        normalized_text = re.sub(r"[^a-z0-9.]+", " ", text.lower()).strip()
        score = 0
        if normalized_entity and normalized_entity in normalized_text:
            score += 10
        if visual_type and visual_type in normalized_text:
            score += 2
        if SOURCE_NOTE_PATTERN.match(text):
            score += 1
        if re.search(r"\b(?:figure|table|chart|diagram|image|map)\s+\d", text, flags=re.IGNORECASE):
            score += 3
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_index >= 0 else 0


def _context_window(blocks: list[dict[str, object]], anchor_index: int) -> list[str]:
    if not blocks:
        return []
    start = max(0, anchor_index - 2)
    end = min(len(blocks), anchor_index + 3)
    selected: list[str] = []
    for block in blocks[start:end]:
        text = _clean_text(block.get("text"))
        if not text or BOILERPLATE_PATTERN.search(text):
            continue
        selected.append(text)
    if not selected and blocks:
        selected.append(_clean_text(blocks[anchor_index].get("text")))
    return selected


def _caption_from_context(context_blocks: list[str], entity_id: str) -> str:
    for text in context_blocks:
        if entity_id.replace("_", " ").lower() in text.lower():
            return text
    for text in context_blocks:
        if re.search(r"\b(?:figure|table|chart|diagram|image|map)\s+\d", text, flags=re.IGNORECASE):
            return text
    return context_blocks[0] if context_blocks else ""


def _footer_from_context(context_blocks: list[str]) -> str:
    for text in reversed(context_blocks):
        if SOURCE_NOTE_PATTERN.match(text):
            return text
    for text in reversed(context_blocks):
        if re.search(r"\b(?:source|note|notes)\b", text, flags=re.IGNORECASE):
            return text
    return context_blocks[-1] if context_blocks else ""


def _resolve_source_pdf(pdf_dir: Path, image_path: Path, explicit_pdf: Path | None = None) -> Path:
    if explicit_pdf is not None:
        if not explicit_pdf.exists():
            raise FileNotFoundError(f"Explicit PDF path does not exist: {explicit_pdf}")
        return explicit_pdf

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
    if len(pdfs) == 1:
        return pdfs[0]

    stem = image_path.stem.lower()
    for pdf in pdfs:
        pdf_stem = re.sub(r"\s+", "", pdf.stem.lower())
        if pdf_stem and pdf_stem in stem.replace(" ", ""):
            return pdf
    return pdfs[0]


def _load_page_context(pdf_path: Path, page_no: int | None, entity_id: str, visual_type: str) -> tuple[str, str, str]:
    if page_no is None:
        return "", "", ""
    if page_no < 1:
        return "", "", ""
    with fitz.open(str(pdf_path)) as doc:
        if page_no > len(doc):
            return "", "", ""
        page = doc[page_no - 1]
        blocks = _extract_pdf_blocks(page)
        if not blocks:
            return "", "", ""
        anchor_index = _find_anchor_index(blocks, entity_id, visual_type)
        context_blocks = _context_window(blocks, anchor_index)
        nearby_context_paragraphs = "\n\n".join(context_blocks)
        chart_heading = _caption_from_context(context_blocks, entity_id)
        chart_footer = _footer_from_context(context_blocks)
        return nearby_context_paragraphs, chart_heading, chart_footer


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _image_files(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.rglob("*")):
        if _is_processable_image(path):
            yield path


def _build_transcription(
    *,
    image_path: Path,
    source_pdf: Path,
    page_no: int | None,
    nearby_context_paragraphs: str,
    chart_heading: str,
    chart_footer: str,
    qwen_analysis: str,
    raw_ocr_literals: str,
) -> UnifiedVisualTranscription:
    entity_id = _infer_entity_id_from_name(image_path)
    visual_type = _infer_visual_type(image_path, entity_id)
    content = (
        f"[IMAGE LOCAL CONTEXT]: {nearby_context_paragraphs.strip()}\n"
        f"[DETAILED DATA METRICS]: {raw_ocr_literals.strip()}\n"
        f"[COMPREHENSIVE ANALYSIS]: {qwen_analysis.strip()}"
    ).strip()
    metadata = {
        "asset_type": "chart_diagram_table",
        "source_file": source_pdf.name,
        "source_path": str(source_pdf),
        "original_pdf_name": source_pdf.name,
        "page_number": page_no,
        "chart_heading": chart_heading,
        "chart_footer": chart_footer,
        "image_path": str(image_path.resolve()),
        "entity_id": entity_id,
        "entity_type": visual_type,
        "source_image_name": image_path.name,
        "transcription_cache_version": "unified_visual_v1",
    }
    return UnifiedVisualTranscription(
        chunk_id=f"visual::{image_path.stem}",
        page_content=content,
        metadata=metadata,
        nearby_context_paragraphs=nearby_context_paragraphs,
        raw_ocr_literals=raw_ocr_literals,
        qwen_visual_analysis=qwen_analysis,
    )


def process_visual_assets(
    *,
    pdf_dir: Path,
    image_dir: Path,
    output_dir: Path,
    model_name: str,
    log_level: str = "INFO",
) -> list[Path]:
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    settings = IngestionSettings()
    ocr = PaddleOcrExtractor(settings, cache_dir=output_dir / "_ocr_cache")
    captioner = QwenVisionCaptioner(settings, model_name=model_name, cache_dir=output_dir / "_qwen_cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(_image_files(image_dir))
    if not image_paths:
        logger.warning("No processable images found in %s", image_dir)
        return []

    print(f"Visual transcription output: {output_dir.resolve()}", flush=True)
    print(f"Found {len(image_paths)} image(s) to process", flush=True)

    written_files: list[Path] = []
    for index, image_path in enumerate(image_paths, start=1):
        try:
            source_pdf = _resolve_source_pdf(pdf_dir, image_path)
            page_no = _page_no_from_path(image_path)
            nearby_context_paragraphs, chart_heading, chart_footer = _load_page_context(
                source_pdf, page_no, _infer_entity_id_from_name(image_path), _infer_visual_type(image_path, _infer_entity_id_from_name(image_path))
            )
            image = ExtractedImage(
                image_path=image_path.resolve(),
                page=page_no,
                type=_infer_visual_type(image_path, _infer_entity_id_from_name(image_path)),
                source_path=str(source_pdf),
                element_id=_infer_entity_id_from_name(image_path),
                metadata={
                    "source_file": source_pdf.name,
                    "page_number": page_no,
                    "entity_id": _infer_entity_id_from_name(image_path),
                    "visual_type": _infer_visual_type(image_path, _infer_entity_id_from_name(image_path)),
                },
            )

            print(f"[{index}/{len(image_paths)}] {image_path.name} -> OCR", flush=True)
            raw_ocr_text = ocr.extract_text(image_path)
            raw_ocr_literals = _literal_csv_text(raw_ocr_text)

            print(f"[{index}/{len(image_paths)}] {image_path.name} -> Qwen", flush=True)
            qwen_visual_analysis, _ = captioner.analyze_image(
                image,
                raw_ocr_literals=raw_ocr_literals,
                nearby_context_paragraphs=nearby_context_paragraphs,
            )

            transcription = _build_transcription(
                image_path=image_path,
                source_pdf=source_pdf,
                page_no=page_no,
                nearby_context_paragraphs=nearby_context_paragraphs,
                chart_heading=chart_heading,
                chart_footer=chart_footer,
                qwen_analysis=qwen_visual_analysis,
                raw_ocr_literals=raw_ocr_literals,
            )
            output_path = output_dir / f"{image_path.stem}.json"
            _write_json(output_path, asdict(transcription))
            written_files.append(output_path)
            print(f"[{index}/{len(image_paths)}] wrote {output_path.name}", flush=True)
        except Exception as exc:
            logger.exception("Failed to transcribe %s: %s", image_path, exc)
            failure_path = output_dir / f"{image_path.stem}.json"
            payload = {
                "chunk_id": f"visual::{image_path.stem}",
                "page_content": "",
                "metadata": {
                    "asset_type": "chart_diagram_table",
                    "source_file": "",
                    "source_path": "",
                    "original_pdf_name": "",
                    "page_number": _page_no_from_path(image_path),
                    "chart_heading": "",
                    "chart_footer": "",
                    "image_path": str(image_path.resolve()),
                    "entity_id": _infer_entity_id_from_name(image_path),
                    "entity_type": _infer_visual_type(image_path, _infer_entity_id_from_name(image_path)),
                    "transcription_cache_version": "unified_visual_v1",
                    "error": str(exc),
                },
                "nearby_context_paragraphs": "",
                "raw_ocr_literals": "",
                "qwen_visual_analysis": "",
            }
            _write_json(failure_path, payload)
            written_files.append(failure_path)
    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified single-chunk visual transcriptions for chart, diagram, and table images."
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Directory containing source PDF files.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing extracted visual images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where unified transcription JSON files will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        help="Qwen2.5-VL model name or local path.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    process_visual_assets(
        pdf_dir=args.pdf_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
