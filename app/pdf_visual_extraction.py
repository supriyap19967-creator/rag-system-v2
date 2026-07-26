import logging
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from langchain_core.documents import Document

from app.ingestion import DEFAULT_PDF_DIR
from app.utils import log_event


logger = logging.getLogger(__name__)

DEFAULT_VISUAL_DIR = Path("assets/extracted_images")
START_PAGE = int(os.getenv("PDF_VISUAL_START_PAGE", "60"))
END_PAGE = int(os.getenv("PDF_VISUAL_END_PAGE", "400"))
GEMINI_MODEL_NAME = (
    os.getenv("GEMINI_VISION_MODEL")
    or os.getenv("GEMINI_CAPTION_MODEL")
    or "gemini-2.0-flash"
)
GEMINI_SYSTEM_PROMPT = (
    "You are a precise technical document parser. Your task is to extract EVERYTHING from the provided image with absolute accuracy. \n"
    "- TEXT EXTRACTION: Transcribe all visible titles, subtitles, headers, data labels, and footnotes verbatim. Do not summarize.\n"
    "- STRUCTURED TABLES: If a table is present, reconstruct it fully in clear Markdown format, ensuring column headers match perfectly.\n"
    "- VISUALS & CHARTS: If a chart or diagram is present, explicitly list the chart type, exact axis titles, intervals, data points, legend keys, and any trends shown. \n"
    "- Give an exhaustive, complete transcription. Do not omit any data points or truncate long descriptions."
)
VISUAL_CATEGORIES = {"Image", "FigureCaption", "Table"}
TEXT_CATEGORIES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "UncategorizedText",
    "CompositeElement",
}
VISUAL_TYPE_KEYWORDS = {
    "chart": ("chart", "graph", "axis", "trend", "plot"),
    "table": ("table", "tabular", "rows", "columns"),
    "figure": ("figure", "diagram", "image", "panel"),
}
BOILERPLATE_PATTERN = re.compile(
    r"\b(?:http|www\.|reproducibility|replication|github|bibliography|references|copyright|doi|isbn|issn)\b",
    re.IGNORECASE,
)
CAPTION_PATTERN = re.compile(r"\b(?:Figure|Fig\.?|Table|Chart|Panel)\s+\d+(?:\.\d+)?[A-Za-z]?", re.IGNORECASE)
CAPTION_TEXT_PATTERN = re.compile(
    r"\b((?:Figure|Fig\.?|Table|Chart|Panel)\s+\d+(?:\.\d+)?[A-Za-z]?\s*[:.\-]?\s*[^|]{12,260})",
    re.IGNORECASE,
)
CAPTION_PREFIX_PATTERN = re.compile(
    r"\b(Fig\.?|Figure|Table|Chart|Panel)\s+(\d+(?:\.\d+)?[A-Za-z]?)\s*[:.\-]?\s*",
    re.IGNORECASE,
)
CAPTION_MIN_LENGTH = int(os.getenv("PDF_VISUAL_CAPTION_MIN_LENGTH", "12"))
CAPTION_MAX_LENGTH = int(os.getenv("PDF_VISUAL_CAPTION_MAX_LENGTH", "220"))
RIGHT_TEXT_DENSITY_THRESHOLD = float(os.getenv("PDF_VISUAL_RIGHT_TEXT_DENSITY_THRESHOLD", "0.028"))
TEXT_ROW_DENSITY_THRESHOLD = float(os.getenv("PDF_VISUAL_TEXT_ROW_DENSITY_THRESHOLD", "0.10"))
TEXT_COLUMN_DENSITY_THRESHOLD = float(os.getenv("PDF_VISUAL_TEXT_COLUMN_DENSITY_THRESHOLD", "0.05"))
MIN_CROP_WIDTH = int(os.getenv("PDF_VISUAL_MIN_CROP_WIDTH", "180"))
MIN_CROP_HEIGHT = int(os.getenv("PDF_VISUAL_MIN_CROP_HEIGHT", "120"))
MAX_CROP_ASPECT_RATIO = float(os.getenv("PDF_VISUAL_MAX_CROP_ASPECT_RATIO", "5.5"))
VISUAL_CROP_DEBUG_PATH = Path(os.getenv("PDF_VISUAL_CROP_DEBUG_PATH", "Data/visual_crop_debug.jsonl"))
MIN_CROP_QUALITY_SCORE = float(os.getenv("PDF_VISUAL_MIN_CROP_QUALITY_SCORE", "0.42"))


@dataclass(frozen=True)
class LayoutElement:
    text: str
    category: str
    page_number: Optional[int]
    image_path: str = ""
    html: str = ""
    crop_quality: str = ""
    extraction_pass: str = ""
    raw_image_path: str = ""
    final_image_path: str = ""
    extraction_method: str = ""
    raw_box: str = ""
    final_box: str = ""
    crop_quality_score: float = 0.0
    rejected_reason: str = ""


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _full_sentences(value: object, max_sentences: int = 3) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;,.")
    sentences = re.findall(r"[^.!?]+[.!?]", text)
    if sentences:
        return " ".join(sentence.strip() for sentence in sentences[:max_sentences])
    words = text.split()
    if len(words) <= 36:
        return text
    return " ".join(words[:36]).strip(" -:;,.")


def _caption_similarity_key(caption: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", caption.lower()).strip()


def _figure_id_from_caption(caption: str, visual_type: str = "visual", page_number: Optional[int] = None) -> str:
    match = CAPTION_PATTERN.search(caption or "")
    if match:
        raw = match.group(0)
        prefix_match = re.search(r"\b(Fig\.?|Figure|Table|Chart|Panel)\s+(\d+(?:\.\d+)?[A-Za-z]?)", raw, re.IGNORECASE)
        if prefix_match:
            kind, number = prefix_match.groups()
            kind = "Figure" if kind.lower().startswith("fig") else kind.title()
            return f"{kind} {number}"
    page = page_number if page_number is not None else "unknown"
    return f"{(visual_type or 'visual').title()} page {page}"


def _section_from_figure_id(figure_id: str) -> str:
    match = re.search(r"\b(?:Figure|Table|Chart|Panel)\s+(\d+)(?:\.\d+)?", figure_id or "", re.IGNORECASE)
    if not match:
        return ""
    return f"Chapter {match.group(1)}"


def _section_for_visual(elements: Sequence["LayoutElement"], visual_index: int, page_number: Optional[int]) -> str:
    for index in range(visual_index - 1, -1, -1):
        element = elements[index]
        if page_number is not None and element.page_number not in (None, page_number):
            continue
        if element.category == "Title":
            text = _clean_text(element.text)
            if text and not CAPTION_PATTERN.search(text) and not BOILERPLATE_PATTERN.search(text):
                return text[:180]
    return ""


def _clean_caption_text(value: object, fallback: str = "") -> str:
    text = _clean_text(value)
    if not text:
        text = _clean_text(fallback)
    if not text:
        return ""

    text = (
        text.replace("\u00ad", "")
        .replace("Â\xad", "")
        .replace("Â", "")
        .replace("â€”", "-")
        .replace("â€“", "-")
    )
    text = re.sub(r"\b(Figure|Table|Chart|Panel)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bFig\.\s+Fig\.\b", "Fig.", text, flags=re.IGNORECASE)
    text = re.sub(r"\bFIGURE\b", "Figure", text)
    text = re.sub(r"\bTABLE\b", "Table", text)
    text = re.sub(r"\bCHART\b", "Chart", text)
    text = re.sub(r"\bFig\.\b", "Figure", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"(?:\s*[−-]?\d+(?:\.\d+)?\s*){3,}$", "", text)
    text = re.sub(r"\b(?:Source|Sources|Note|Notes)\s*:\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -;,.")

    prefix_match = CAPTION_PREFIX_PATTERN.search(text)
    if prefix_match:
        raw_kind, number = prefix_match.groups()
        kind = "Figure" if raw_kind.lower().startswith("fig") else raw_kind.title()
        title_start = prefix_match.end()
        title = text[title_start:].strip(" :-")
        title = re.sub(r"\bLimited access to credit,\s*managerial know-how\s*", "", title, flags=re.IGNORECASE)
        title = re.split(
            r"\s+(?:Source|Sources|Note|Notes|This figure|The figure|The report)\b",
            title,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        title = re.sub(r"\bthan do firms\b", "than firms", title, flags=re.IGNORECASE)
        text = f"{kind} {number}: {title}".strip()

    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    words = text.split()
    for size in range(4, 11):
        if len(words) >= size * 2 and words[-size:] == words[-2 * size : -size]:
            words = words[:-size]
            text = " ".join(words)
            break
    text = re.sub(r"[|]{2,}", "|", text)
    text = re.sub(r"\s+(?:[A-Za-z]{1,2}|[−-]?\d+(?:\.\d+)?)(?:\s+[A-Za-z]{1,2}|[−-]?\d+(?:\.\d+)?){2,}$", "", text)
    text = re.sub(r"\s+[A-Za-z]{1,4}$", "", text) if len(text.split()) > 8 else text
    text = re.sub(r"[-:;,.]{2,}$", "", text).strip(" -;,.")

    if len(text) > CAPTION_MAX_LENGTH:
        boundary = max(text.rfind(".", 0, CAPTION_MAX_LENGTH), text.rfind(";", 0, CAPTION_MAX_LENGTH))
        if boundary >= CAPTION_MIN_LENGTH:
            text = text[: boundary + 1]
        else:
            text = text[:CAPTION_MAX_LENGTH].rsplit(" ", 1)[0]
    if text and text[-1].isalnum():
        text += "."
    if len(text) < CAPTION_MIN_LENGTH:
        return ""
    return text


def _element_category(element: object) -> str:
    category = getattr(element, "category", "") or element.__class__.__name__
    return str(category or "")


def _element_metadata(element: object) -> object:
    return getattr(element, "metadata", None)


def _metadata_value(metadata: object, key: str) -> object:
    if metadata is None:
        return None
    if hasattr(metadata, key):
        return getattr(metadata, key)
    if isinstance(metadata, dict):
        return metadata.get(key)
    try:
        return metadata.to_dict().get(key)
    except Exception:
        return None


def _layout_element(element: object) -> LayoutElement:
    metadata = _element_metadata(element)
    html = _metadata_value(metadata, "text_as_html")
    image_path = (
        _metadata_value(metadata, "image_path")
        or _metadata_value(metadata, "image_filename")
        or ""
    )
    return LayoutElement(
        text=_clean_text(str(element)),
        category=_element_category(element),
        page_number=_metadata_value(metadata, "page_number"),
        image_path=str(image_path or ""),
        html=str(html or ""),
    )


def _selected_page_pdf(pdf_path: Path, start_page: int, end_page: int) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory()
    selected_pdf_path = Path(temp_dir.name) / f"{pdf_path.stem}-pages-{start_page}-{end_page}.pdf"
    try:
        import fitz

        source = fitz.open(str(pdf_path))
        selected = fitz.open()
        first_index = max(start_page - 1, 0)
        last_index = min(end_page - 1, len(source) - 1)
        if first_index <= last_index:
            selected.insert_pdf(source, from_page=first_index, to_page=last_index)
            selected.save(str(selected_pdf_path))
        selected.close()
        source.close()
    except Exception as exc:
        temp_dir.cleanup()
        raise RuntimeError(f"Could not create selected-page PDF for {pdf_path}: {exc}") from exc
    return selected_pdf_path, temp_dir


def _remap_page_number(page_number: Optional[int], start_page: int) -> Optional[int]:
    if page_number is None:
        return None
    try:
        return int(page_number) + start_page - 1
    except (TypeError, ValueError):
        return page_number


def _box_to_json(box: object) -> str:
    if not box:
        return ""
    try:
        values = [float(box.x0), float(box.y0), float(box.x1), float(box.y1)]
    except AttributeError:
        try:
            values = [float(value) for value in box]
        except Exception:
            return ""
    return json.dumps([round(value, 2) for value in values], separators=(",", ":"))


def _box_from_json(value: str) -> List[float]:
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    if isinstance(parsed, list) and len(parsed) == 4:
        try:
            return [float(item) for item in parsed]
        except Exception:
            return []
    return []


def _write_crop_debug(record: Dict[str, object]) -> None:
    try:
        VISUAL_CROP_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with VISUAL_CROP_DEBUG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("Could not write visual crop debug record: %s", exc)


def _partition_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    start_page: int = START_PAGE,
    end_page: int = END_PAGE,
) -> List[LayoutElement]:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        logger.warning("Unstructured PDF extraction skipped because unstructured[pdf] is not installed: %s", exc)
        return []

    print(f"--- Extracted images will be saved into: {output_dir.resolve()} ---", flush=True)
    for page_number in range(start_page, end_page + 1):
        print(f"--- Starting Partitioning for Page {page_number} ---", flush=True)

    selected_pdf_path, temp_dir = _selected_page_pdf(pdf_path, start_page, end_page)
    try:
        try:
            raw_elements = partition_pdf(
                filename=str(selected_pdf_path),
                strategy="hi_res",
                extract_images_to_dir=str(output_dir),
                extract_images_in_pdf=True,
                extract_image_block_types=["Image", "Table"],
                infer_table_structure=True,
                starting_page_number=start_page,
            )
        except TypeError:
            try:
                raw_elements = partition_pdf(
                    filename=str(selected_pdf_path),
                    strategy="hi_res",
                    extract_images_to_dir=str(output_dir),
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table"],
                    infer_table_structure=True,
                )
            except TypeError:
                try:
                    raw_elements = partition_pdf(
                        filename=str(selected_pdf_path),
                        strategy="hi_res",
                        extract_image_block_output_dir=str(output_dir),
                        extract_image_block_types=["Image", "Table"],
                        extract_image_block_to_payload=False,
                        infer_table_structure=True,
                    )
                except TypeError:
                    raw_elements = partition_pdf(
                        filename=str(selected_pdf_path),
                        strategy="hi_res",
                        infer_table_structure=True,
                    )
    except Exception as exc:
        logger.warning("Unstructured PDF extraction failed for %s: %s", pdf_path, exc)
        return _fallback_pymupdf_visual_elements(pdf_path, output_dir, start_page=start_page, end_page=end_page)
    finally:
        temp_dir.cleanup()

    elements = [_layout_element(element) for element in raw_elements]
    remapped: List[LayoutElement] = []
    for element in elements:
        remapped.append(
            LayoutElement(
                text=element.text,
                category=element.category,
                page_number=_remap_page_number(element.page_number, start_page),
                image_path=element.image_path,
                html=element.html,
                crop_quality=element.crop_quality,
                extraction_pass=element.extraction_pass,
                raw_image_path=element.raw_image_path,
                final_image_path=element.final_image_path,
                extraction_method=element.extraction_method,
                raw_box=element.raw_box,
                final_box=element.final_box,
                crop_quality_score=element.crop_quality_score,
                rejected_reason=element.rejected_reason,
            )
        )
    return remapped


def _extract_page_text_lines(page: object) -> List[Dict[str, object]]:
    lines: List[Dict[str, object]] = []
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return lines
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = _clean_text(" ".join(str(span.get("text") or "") for span in spans))
            if not text:
                continue
            bbox_values = line.get("bbox") or block.get("bbox")
            if not bbox_values:
                continue
            lines.append({"text": text, "bbox": tuple(float(value) for value in bbox_values)})
    return lines


def _caption_candidates_from_lines(lines: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for index, line in enumerate(lines):
        text = str(line.get("text") or "")
        if not CAPTION_PATTERN.search(text):
            continue
        caption_parts = [text]
        bbox = list(line.get("bbox") or (0, 0, 0, 0))
        for next_line in lines[index + 1 : index + 4]:
            next_text = str(next_line.get("text") or "")
            if CAPTION_PATTERN.search(next_text) or BOILERPLATE_PATTERN.search(next_text):
                break
            if len(" ".join(caption_parts)) > CAPTION_MAX_LENGTH:
                break
            next_bbox = list(next_line.get("bbox") or bbox)
            if abs(float(next_bbox[0]) - float(bbox[0])) > 28:
                break
            caption_parts.append(next_text)
            bbox = [
                min(float(bbox[0]), float(next_bbox[0])),
                min(float(bbox[1]), float(next_bbox[1])),
                max(float(bbox[2]), float(next_bbox[2])),
                max(float(bbox[3]), float(next_bbox[3])),
            ]
        caption = _clean_caption_text(" ".join(caption_parts))
        if caption:
            candidates.append({"caption": caption, "bbox": tuple(bbox)})
    return candidates


CHART_SIGNAL_PATTERN = re.compile(
    r"\b(?:sales|%|percent|log scale|axis|standard adopted|no standard adopted|legend|x-axis|y-axis)\b",
    re.IGNORECASE,
)


def _visual_type_from_caption(caption: str) -> str:
    lowered = (caption or "").lower()
    if lowered.startswith("table") or " table " in f" {lowered} ":
        return "table"
    if any(term in lowered for term in ("chart", "graph", "axis", "trend", "sales", "percent", "%")):
        return "chart"
    return "figure"


def _line_text(value: Dict[str, object]) -> str:
    return str(value.get("text") or "")


def _is_multi_panel_figure(caption: str, lines: Sequence[Dict[str, object]], caption_bbox: Sequence[float], page_width: float, page_height: float) -> bool:
    if not re.search(r"\bFigure\s+\d+(?:\.\d+)?", caption or "", re.IGNORECASE):
        return False
    _x0, caption_top, _x1, caption_bottom = [float(value) for value in caption_bbox]
    panel_search_top = min(caption_top + page_height * 0.025, caption_bottom)
    panel_search_bottom = caption_top + page_height * 0.24
    panel_labels = 0
    for line in lines:
        bbox = line.get("bbox")
        if not bbox:
            continue
        lx0, ly0, _lx1, _ly1 = [float(value) for value in bbox]
        if ly0 < panel_search_top or ly0 > panel_search_bottom:
            continue
        text = _line_text(line)
        if re.match(r"^\s*[a-d]\.\s+\S+", text, flags=re.IGNORECASE):
            panel_labels += 1
        elif re.match(r"^\s*[a-d]\.\s*$", text, flags=re.IGNORECASE):
            panel_labels += 1
        elif panel_labels and lx0 > page_width * 0.52 and re.match(r"^\s*[a-d]\b", text, flags=re.IGNORECASE):
            panel_labels += 1
    return panel_labels >= 2


def _multi_panel_crop_bottom(lines: Sequence[Dict[str, object]], caption_bbox: Sequence[float], page_height: float) -> float:
    _x0, caption_top, _x1, caption_bottom = [float(value) for value in caption_bbox]
    bottom = min(page_height, caption_top + page_height * 0.52)
    for line in lines:
        bbox = line.get("bbox")
        if not bbox:
            continue
        _lx0, ly0, _lx1, _ly1 = [float(value) for value in bbox]
        if ly0 <= caption_top + page_height * 0.16:
            continue
        text = _line_text(line).strip()
        if re.match(r"^(Sources?|Notes?)\s*:", text, flags=re.IGNORECASE):
            return max(caption_top + page_height * 0.28, ly0 - page_height * 0.004)
    return bottom


def _table_crop_bottom(lines: Sequence[Dict[str, object]], caption_bbox: Sequence[float], page_height: float) -> float:
    _x0, caption_top, _x1, caption_bottom = [float(value) for value in caption_bbox]
    min_table_bottom = min(page_height, caption_bottom + page_height * 0.10)
    source_search_top = min(page_height, caption_bottom + page_height * 0.055)
    for line in lines:
        bbox = line.get("bbox")
        if not bbox:
            continue
        _lx0, ly0, _lx1, _ly1 = [float(value) for value in bbox]
        if ly0 <= source_search_top:
            continue
        text = _line_text(line).strip()
        if re.match(r"^(Sources?|Notes?)\s*:", text, flags=re.IGNORECASE):
            return max(min_table_bottom, ly0 - page_height * 0.008)
        if re.match(r"^(Figure|Table)\s+\d+(?:\.\d+)?", text, flags=re.IGNORECASE):
            return max(min_table_bottom, ly0 - page_height * 0.03)
    return page_height * 0.965


def _rect_area(rect: object) -> float:
    try:
        return max(0.0, float(rect.x1) - float(rect.x0)) * max(0.0, float(rect.y1) - float(rect.y0))
    except Exception:
        return 0.0


def _intersects_column(rect: object, left: float, right: float) -> bool:
    overlap = max(0.0, min(float(rect.x1), right) - max(float(rect.x0), left))
    return overlap >= min((float(rect.x1) - float(rect.x0)) * 0.35, (right - left) * 0.35)


def _visual_regions_from_page(page: object, *, left: float, right: float, caption_bottom: float) -> List[object]:
    """Return drawing/image regions near the caption column.

    This is intentionally conservative: paragraph text is ignored, and only PDF
    drawings or embedded-image blocks are allowed to seed the visual crop.
    """
    import fitz

    page_rect = page.rect
    max_bottom = min(float(page_rect.height), caption_bottom + float(page_rect.height) * 0.46)
    regions: List[object] = []
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if not rect:
                continue
            rect = fitz.Rect(rect)
            if rect.y1 < caption_bottom - float(page_rect.height) * 0.12 or rect.y0 > max_bottom:
                continue
            if not _intersects_column(rect, left, right):
                continue
            if _rect_area(rect) < 35:
                continue
            regions.append(rect)
    except Exception:
        pass

    try:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 1 or not block.get("bbox"):
                continue
            rect = fitz.Rect(block["bbox"])
            if rect.y1 < caption_bottom - float(page_rect.height) * 0.12 or rect.y0 > max_bottom:
                continue
            if _intersects_column(rect, left, right) and _rect_area(rect) >= 700:
                regions.append(rect)
    except Exception:
        pass
    return regions


def _union_rect(rects: Sequence[object]) -> Optional[object]:
    if not rects:
        return None
    import fitz

    union = fitz.Rect(rects[0])
    for rect in rects[1:]:
        union |= fitz.Rect(rect)
    return union


def _detected_visual_region_rect(
    page: object,
    caption_bbox: Sequence[float],
    *,
    visual_type: str,
) -> Optional[object]:
    import fitz

    page_rect = page.rect
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)
    x0, _y0, x1, y1 = [float(value) for value in caption_bbox]
    caption_center_x = (x0 + x1) / 2
    if caption_center_x <= page_width / 2:
        left, right = 0.0, page_width * 0.56
    else:
        left, right = page_width * 0.44, page_width

    regions = _visual_regions_from_page(page, left=left, right=right, caption_bottom=y1)
    if not regions:
        return None

    if visual_type == "table":
        usable = [
            rect
            for rect in regions
            if float(rect.width) >= page_width * 0.12 and float(rect.height) >= page_height * 0.025
        ]
    elif visual_type == "chart":
        usable = [
            rect
            for rect in regions
            if float(rect.width) >= page_width * 0.08 or float(rect.height) >= page_height * 0.035
        ]
    else:
        usable = [rect for rect in regions if _rect_area(rect) >= 500]
    if not usable:
        usable = regions

    union = _union_rect(usable)
    if not union:
        return None

    # Keep the detected region in the same text column horizontally but allow scaling up to full page width.
    union.x0 = max(0.0, union.x0)
    union.x1 = min(page_width, union.x1)
    union.y0 = max(0, min(union.y0, y1 - page_height * 0.07))
    union.y1 = min(page_height, max(union.y1, y1 + page_height * 0.10))
    return fitz.Rect(union)


def _padded_region_rect(
    page: object,
    region_rect: object,
    caption_bbox: Sequence[float],
    *,
    visual_type: str,
    pass_no: int,
) -> object:
    import fitz

    page_rect = page.rect
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)
    caption_x0, caption_y0, caption_x1, caption_y1 = [float(value) for value in caption_bbox]
    caption_center_x = (caption_x0 + caption_x1) / 2
    if caption_center_x <= page_width / 2:
        column_left, column_right = 0.0, page_width * 0.56
    else:
        column_left, column_right = page_width * 0.44, page_width

    if visual_type == "chart":
        pad_left, pad_right = page_width * 0.035, page_width * 0.035
        pad_top = page_height * 0.035
        pad_bottom = page_height * (0.16 if pass_no == 1 else 0.24)
    elif visual_type == "table":
        pad_left = pad_right = page_width * (0.04 if pass_no == 1 else 0.06)
        pad_top = page_height * (0.035 if pass_no == 1 else 0.055)
        pad_bottom = page_height * (0.055 if pass_no == 1 else 0.085)
    else:
        pad_left = pad_right = page_width * (0.055 if pass_no == 1 else 0.075)
        pad_top = page_height * 0.04
        pad_bottom = page_height * (0.07 if pass_no == 1 else 0.10)

    rect = fitz.Rect(
        max(0.0, min(float(region_rect.x0), caption_x0) - pad_left),
        max(0.0, min(float(region_rect.y0), caption_y0) - pad_top),
        min(page_width, max(float(region_rect.x1), caption_x1) + pad_right),
        min(page_height, max(float(region_rect.y1), caption_y1) + pad_bottom),
    )
    return rect


def _render_page_crop(
    page: object,
    crop_rect: object,
    output_path: Path,
    *,
    trim_right_text: bool = True,
) -> Optional[Dict[str, object]]:
    try:
        import fitz

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect, alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output_path.with_name(f"{output_path.stem}.raw{output_path.suffix}")
        if raw_path.exists():
            try:
                raw_path.unlink()
            except Exception:
                pass
        pix.save(str(output_path))
        _trim_visual_crop(output_path, trim_right_text=trim_right_text)
        return {
            "crop_quality": "unknown",
            "extraction_pass": "pass_1",
            "crop_box": [
                round(float(crop_rect.x0), 2),
                round(float(crop_rect.y0), 2),
                round(float(crop_rect.x1), 2),
                round(float(crop_rect.y1), 2),
            ],
        }
    except Exception as exc:
        logger.warning("PyMuPDF fallback crop failed for %s: %s", output_path, exc)
        return None


def _line_overlaps_column(line_bbox: Sequence[float], left: float, right: float) -> bool:
    x0, _y0, x1, _y1 = [float(value) for value in line_bbox]
    overlap = max(0.0, min(x1, right) - max(x0, left))
    return overlap >= min((x1 - x0) * 0.45, (right - left) * 0.25)


def _chart_signal_bottom(
    lines: Sequence[Dict[str, object]],
    *,
    caption_bbox: Sequence[float],
    left: float,
    right: float,
    page_height: float,
) -> Optional[float]:
    _x0, _y0, _x1, caption_bottom = [float(value) for value in caption_bbox]
    bottom = None
    max_signal_y = caption_bottom + page_height * 0.38
    for line in lines:
        text = str(line.get("text") or "")
        bbox = line.get("bbox")
        if not bbox:
            continue
        lx0, ly0, lx1, ly1 = [float(value) for value in bbox]
        if ly0 < caption_bottom or ly0 > max_signal_y:
            continue
        if not _line_overlaps_column((lx0, ly0, lx1, ly1), left, right):
            continue
        if CHART_SIGNAL_PATTERN.search(text) or re.search(r"\b\d{1,3}(?:\.\d+)?\b", text):
            bottom = max(bottom or ly1, ly1)
    if bottom is None:
        return None
    return min(page_height, bottom + page_height * 0.055)


def _is_chart_caption(caption: str, lines: Sequence[Dict[str, object]], caption_bbox: Sequence[float], left: float, right: float, page_height: float) -> bool:
    if not re.search(r"\bFigure\s+\d+(?:\.\d+)?", caption, re.IGNORECASE):
        return False
    if CHART_SIGNAL_PATTERN.search(caption):
        return True
    return _chart_signal_bottom(lines, caption_bbox=caption_bbox, left=left, right=right, page_height=page_height) is not None


def _fallback_crop_rect(
    page: object,
    caption_bbox: Sequence[float],
    *,
    caption: str = "",
    lines: Sequence[Dict[str, object]] = (),
    force_pass: int = 1,
) -> tuple[object, str, str, str]:
    import fitz

    page_rect = page.rect
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)
    x0, y0, x1, y1 = [float(value) for value in caption_bbox]
    caption_center_x = (x0 + x1) / 2
    if caption_center_x <= page_width / 2:
        left = max(0, min(x0 - page_width * 0.06, page_width * 0.04))
        right = min(page_width * 0.56, max(x1 + page_width * 0.04, page_width * 0.50))
    else:
        left = max(page_width * 0.44, min(x0 - page_width * 0.04, page_width * 0.50))
        right = min(page_width, max(x1 + page_width * 0.06, page_width * 0.96))

    detected_type = _visual_type_from_caption(caption)
    chart_like = detected_type == "chart" or _is_chart_caption(caption, lines, caption_bbox, left, right, page_height)
    if chart_like:
        detected_type = "chart"
    if detected_type == "table":
        table_regions = _visual_regions_from_page(page, left=0.0, right=page_width, caption_bottom=y1)
        usable_table_regions = [
            rect
            for rect in table_regions
            if float(rect.width) >= page_width * 0.12 and float(rect.height) >= page_height * 0.018
        ] or table_regions
        table_region = _union_rect(usable_table_regions)
        table_left = page_width * 0.03
        table_right = page_width * 0.97
        table_top = max(0, y0 - page_height * (0.035 if force_pass == 1 else 0.055))
        table_text_bottom = _table_crop_bottom(lines, caption_bbox, page_height)
        table_has_text_boundary = table_text_bottom < page_height * 0.94
        if table_region:
            table_top = max(0, min(table_top, float(table_region.y0) - page_height * 0.025))
            if table_has_text_boundary:
                table_bottom = min(page_height, table_text_bottom + page_height * (0.004 if force_pass == 1 else 0.012))
            else:
                table_bottom = min(
                    page_height,
                    max(float(table_region.y1), y1, table_text_bottom) + page_height * (0.01 if force_pass == 1 else 0.025),
                )
        else:
            table_bottom = min(page_height, table_text_bottom + page_height * (0.01 if force_pass == 1 else 0.025))
        return (
            fitz.Rect(table_left, table_top, table_right, table_bottom),
            "table_full_width_candidate",
            f"pass_{force_pass}",
            detected_type,
        )
    if _is_multi_panel_figure(caption, lines, caption_bbox, page_width, page_height):
        top = max(0, y0 - page_height * 0.035)
        bottom = _multi_panel_crop_bottom(lines, caption_bbox, page_height)
        if force_pass >= 2:
            top = max(0, top - page_height * 0.015)
            bottom = min(page_height, bottom + page_height * 0.03)
        return (
            fitz.Rect(page_width * 0.035, top, page_width * 0.965, bottom),
            "multi_panel_figure_candidate",
            f"pass_{force_pass}",
            "chart" if chart_like else detected_type,
        )
    region_rect = _detected_visual_region_rect(page, caption_bbox, visual_type=detected_type)
    if region_rect:
        rect = _padded_region_rect(
            page,
            region_rect,
            caption_bbox,
            visual_type=detected_type,
            pass_no=force_pass,
        )
        return rect, f"{detected_type}_layout_region_candidate", f"pass_{force_pass}", detected_type

    if chart_like:
        top = max(0, y0 - page_height * 0.045)
        signal_bottom = _chart_signal_bottom(lines, caption_bbox=caption_bbox, left=left, right=right, page_height=page_height)
        bottom = min(page_height, max(y1 + page_height * 0.34, signal_bottom or 0))
        if force_pass >= 2:
            left = max(0, left - page_width * 0.025)
            right = min(page_width, right + page_width * 0.025)
            bottom = min(page_height, bottom + page_height * 0.10)
        quality = "chart_complete_candidate" if signal_bottom else "chart_low_signal_expanded"
        extraction_pass = f"pass_{force_pass}"
    else:
        top = max(0, y0 - page_height * 0.12)
        bottom = min(page_height, y1 + page_height * 0.18)
        if bottom - top < page_height * 0.20:
            top = max(0, y0 - page_height * 0.20)
        quality = "layout_region_candidate"
        extraction_pass = f"pass_{force_pass}"
    return fitz.Rect(left, top, right, bottom), quality, extraction_pass, detected_type


def _chart_crop_complete(image_path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return True
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if width < MIN_CROP_WIDTH or height < int(MIN_CROP_HEIGHT * 1.45):
                return False
            mask = _dark_pixel_mask(image.convert("RGB"))
            lower_density = _density(mask, 0, int(height * 0.55), width, height)
            left_density = _density(mask, 0, int(height * 0.25), int(width * 0.28), height)
            return lower_density >= 0.004 and left_density >= 0.004
    except Exception:
        return True


def _fallback_pymupdf_visual_elements(
    pdf_path: Path,
    output_dir: Path,
    *,
    start_page: int,
    end_page: int,
) -> List[LayoutElement]:
    try:
        import fitz
    except ImportError as exc:
        logger.warning("PyMuPDF visual fallback skipped because fitz is not installed: %s", exc)
        return []

    elements: List[LayoutElement] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("PyMuPDF visual fallback could not open %s: %s", pdf_path, exc)
        return []

    try:
        for page_number in range(start_page, min(end_page, len(doc)) + 1):
            page = doc[page_number - 1]
            lines = _extract_page_text_lines(page)
            candidates = _caption_candidates_from_lines(lines)
            page_visual_count = 0
            for candidate in candidates:
                caption = str(candidate.get("caption") or "")
                bbox = candidate.get("bbox")
                if not caption or not bbox:
                    continue
                page_visual_count += 1
                visual_type = _visual_type_from_caption(caption)
                image_path = output_dir / _safe_visual_filename(
                    pdf_path,
                    page_visual_count,
                    suffix=".png",
                    page_number=page_number,
                    visual_type=visual_type,
                )
                crop_rect, crop_quality, extraction_pass, visual_type = _fallback_crop_rect(
                    page,
                    bbox,
                    caption=caption,
                    lines=lines,
                    force_pass=1,
                )
                render_meta = _render_page_crop(
                    page,
                    crop_rect,
                    image_path,
                    trim_right_text=(visual_type != "chart" and not crop_quality.startswith(("multi_panel", "table_full_width"))),
                )
                if not render_meta:
                    continue
                quality_result = _quality_score_for_crop(image_path, visual_type)
                chart_like = visual_type == "chart" or crop_quality.startswith("chart")
                log_event(
                    logger,
                    logging.INFO,
                    "visual_crop_completeness_checked",
                    source_pdf=str(pdf_path),
                    page=page_number,
                    caption=caption,
                    image_path=str(image_path),
                    visual_type=visual_type,
                    crop_quality_score=quality_result.get("score"),
                    complete=bool(quality_result.get("usable")),
                    reason=";".join(str(reason) for reason in quality_result.get("reasons", [])),
                )
                if (chart_like and not _chart_crop_complete(image_path)) or not bool(quality_result.get("usable")):
                    log_event(
                        logger,
                        logging.INFO,
                        "visual_recrop_attempted",
                        source_pdf=str(pdf_path),
                        page=page_number,
                        caption=caption,
                        image_path=str(image_path),
                        visual_type=visual_type,
                        pass_used="pass_2",
                        reason=";".join(str(reason) for reason in quality_result.get("reasons", [])),
                    )
                    retry_rect, retry_quality, retry_pass, retry_type = _fallback_crop_rect(
                        page,
                        bbox,
                        caption=caption,
                        lines=lines,
                        force_pass=2,
                    )
                    retry_type = retry_type or visual_type
                    retry_path = image_path.with_name(f"{image_path.stem}.pass2{image_path.suffix}")
                    retry_meta = _render_page_crop(
                        page,
                        retry_rect,
                        retry_path,
                        trim_right_text=(retry_type != "chart" and not retry_quality.startswith(("multi_panel", "table_full_width"))),
                    )
                    if retry_meta:
                        retry_quality_result = _quality_score_for_crop(retry_path, retry_type)
                        retry_score = float(retry_quality_result.get("score") or 0.0)
                        pass1_score = float(quality_result.get("score") or 0.0)
                        if retry_score >= pass1_score:
                            try:
                                shutil.copy2(retry_path, image_path)
                            except Exception:
                                pass
                            crop_rect = retry_rect
                            crop_quality = (
                                f"{retry_quality}_accepted"
                                if bool(retry_quality_result.get("usable"))
                                else f"{retry_quality}_low_quality"
                            )
                            extraction_pass = retry_pass
                            visual_type = retry_type
                            render_meta = retry_meta
                            quality_result = retry_quality_result
                elif chart_like:
                    crop_quality = "chart_complete"

                if bool(quality_result.get("usable")) and "low_quality" not in crop_quality:
                    crop_quality = crop_quality.replace("_candidate", "_accepted")
                if not bool(quality_result.get("usable")):
                    log_event(
                        logger,
                        logging.INFO,
                        "visual_skipped_incomplete",
                        source_pdf=str(pdf_path),
                        page=page_number,
                        caption=caption,
                        image_path=str(image_path),
                        visual_type=visual_type,
                        crop_quality_score=quality_result.get("score"),
                        reason=";".join(str(reason) for reason in quality_result.get("reasons", [])),
                    )

                crop_box = render_meta.get("crop_box") or [
                    round(float(crop_rect.x0), 2),
                    round(float(crop_rect.y0), 2),
                    round(float(crop_rect.x1), 2),
                    round(float(crop_rect.y1), 2),
                ]
                raw_image_path = image_path.with_name(f"{image_path.stem}.raw{image_path.suffix}")
                rejected_reason = ";".join(str(reason) for reason in quality_result.get("reasons", []))
                crop_quality_score = float(quality_result.get("score") or 0.0)
                final_box = _box_to_json(crop_box)
                raw_box = _box_to_json(bbox)
                elements.append(LayoutElement(text=caption, category="FigureCaption", page_number=page_number))
                elements.append(
                    LayoutElement(
                        text=caption,
                        category="Image",
                        page_number=page_number,
                        image_path=str(image_path),
                        crop_quality=crop_quality,
                        extraction_pass=extraction_pass,
                        raw_image_path=str(raw_image_path),
                        final_image_path=str(image_path),
                        extraction_method="pymupdf_page_crop",
                        raw_box=raw_box,
                        final_box=final_box,
                        crop_quality_score=crop_quality_score,
                        rejected_reason="" if bool(quality_result.get("usable")) else rejected_reason,
                    )
                )
                _write_crop_debug(
                    {
                        "source_pdf": str(pdf_path),
                        "page": page_number,
                        "caption": caption,
                        "visual_type": visual_type,
                        "extraction_method": "pymupdf_page_crop",
                        "raw_box": _box_from_json(raw_box),
                        "final_box": _box_from_json(final_box),
                        "crop_quality_score": crop_quality_score,
                        "crop_quality_usable": bool(quality_result.get("usable")),
                        "rejected_reason": "" if bool(quality_result.get("usable")) else rejected_reason,
                        "crop_pass_used": extraction_pass,
                        "raw_crop_path": str(raw_image_path),
                        "final_crop_path": str(image_path),
                        "quality_metrics": quality_result.get("metrics", {}),
                    }
                )
                log_event(
                    logger,
                    logging.INFO,
                    "pdf_visual_pymupdf_fallback_extracted",
                    source_pdf=str(pdf_path),
                    page=page_number,
                    caption=caption,
                    image_path=str(image_path),
                    crop_box=crop_box,
                    crop_quality=crop_quality,
                    crop_quality_score=crop_quality_score,
                    extraction_pass=extraction_pass,
                    raw_image_path=str(raw_image_path),
                    rejected_reason="" if bool(quality_result.get("usable")) else rejected_reason,
                )
    finally:
        doc.close()
    return elements


def _safe_visual_filename(pdf_path: Path, index: int, suffix: str = ".png", page_number: Optional[int] = None, visual_type: str = "chart") -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", pdf_path.stem).strip("-") or "pdf"
    safe_type = re.sub(r"[^A-Za-z0-9_.-]+", "-", visual_type or "chart").strip("-") or "chart"
    if page_number:
        return f"page{page_number}_{safe_type}{index}{suffix}"
    return f"{safe_stem}-visual-{index}{suffix}"


def _safe_visual_entity_filename(pdf_path: Path, figure_id: str, page_number: Optional[int], suffix: str = ".png") -> str:
    safe_doc = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_") or "document"
    safe_entity = re.sub(r"[^a-z0-9]+", "_", str(figure_id or "").lower()).strip("_")
    if not safe_entity:
        safe_entity = f"page_{page_number or 'unknown'}_visual"
    return f"{safe_doc}_{safe_entity}{suffix or '.png'}"


def _canonicalize_visual_image_path(
    image_path: str,
    output_dir: Path,
    pdf_path: Path,
    figure_id: str,
    page_number: Optional[int],
) -> str:
    if not image_path:
        return ""
    source_path = Path(image_path)
    if not source_path.is_absolute():
        source_path = next(
            (
                candidate
                for candidate in (output_dir / source_path, pdf_path.parent / source_path, source_path)
                if candidate.exists()
            ),
            source_path,
        )
    if not source_path.exists():
        logger.warning("Visual image path was not written to disk before metadata binding: %s", source_path)
        return str(source_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / _safe_visual_entity_filename(
        pdf_path,
        figure_id,
        page_number,
        suffix=source_path.suffix or ".png",
    )
    if source_path.resolve() != target_path.resolve():
        try:
            shutil.copy2(source_path, target_path)
        except Exception as exc:
            logger.warning("Could not canonicalize visual image path %s to %s: %s", source_path, target_path, exc)
            return str(source_path)
    print(
        f"VALIDATION [Image Save]: figure_id={figure_id} image_path={target_path} exists={target_path.exists()}",
        flush=True,
    )
    return str(target_path)


def _normalize_image_path(
    image_path: str,
    output_dir: Path,
    pdf_path: Path,
    index: int,
    *,
    page_number: Optional[int] = None,
    visual_type: str = "chart",
) -> str:
    if not image_path:
        return ""

    source_path = Path(image_path)
    if not source_path.is_absolute():
        candidates = [
            output_dir / source_path,
            pdf_path.parent / source_path,
            source_path,
        ]
        source_path = next((candidate for candidate in candidates if candidate.exists()), source_path)

    if not source_path.exists():
        return str(source_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix or ".png"
    target_path = output_dir / _safe_visual_filename(
        pdf_path,
        index,
        suffix=suffix,
        page_number=page_number,
        visual_type=visual_type,
    )
    if source_path.resolve() != target_path.resolve():
        try:
            shutil.copy2(source_path, target_path)
            return str(_trim_visual_crop(target_path, trim_right_text=(visual_type != "chart")))
        except Exception as exc:
            logger.warning("Could not copy extracted image %s to %s: %s", source_path, target_path, exc)
    return str(_trim_visual_crop(source_path, trim_right_text=(visual_type != "chart")))


def _dark_pixel_mask(image: object) -> List[List[bool]]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    return [[pixels[x, y] < 225 for x in range(width)] for y in range(height)]


def _density(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> float:
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    dark = 0
    for y in range(max(top, 0), min(bottom, len(mask))):
        row = mask[y]
        for x in range(max(left, 0), min(right, len(row))):
            if row[x]:
                dark += 1
    return dark / float(width * height)


def _row_text_like_ratio(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> float:
    rows = 0
    text_like = 0
    for y in range(max(top, 0), min(bottom, len(mask))):
        rows += 1
        row = mask[y]
        row_density = sum(1 for x in range(max(left, 0), min(right, len(row))) if row[x]) / max(right - left, 1)
        if TEXT_ROW_DENSITY_THRESHOLD <= row_density <= 0.55:
            text_like += 1
    return text_like / max(rows, 1)


def _right_strip_is_text_heavy(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> bool:
    width = max(right - left, 1)
    strip_left = int(right - width * 0.20)
    strip_density = _density(mask, strip_left, top, right, bottom)
    row_ratio = _row_text_like_ratio(mask, strip_left, top, right, bottom)
    return strip_density >= RIGHT_TEXT_DENSITY_THRESHOLD and row_ratio >= TEXT_COLUMN_DENSITY_THRESHOLD


def _column_gap_trim_right(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> Optional[int]:
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    search_start = left + int(width * 0.35)
    search_end = left + int(width * 0.78)
    min_gap = max(18, int(width * 0.035))
    gap_start: Optional[int] = None
    for x in range(search_start, min(search_end, right)):
        dark = 0
        for y in range(max(top, 0), min(bottom, len(mask))):
            if mask[y][x]:
                dark += 1
        column_density = dark / float(height)
        if column_density <= 0.01:
            if gap_start is None:
                gap_start = x
            if x - gap_start + 1 >= min_gap:
                candidate_right = max(left + MIN_CROP_WIDTH, gap_start + 8)
                if candidate_right < right - 20:
                    return candidate_right
        else:
            gap_start = None
    return None


def _column_gap_trim_left(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> Optional[int]:
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    band_top = top + int(height * 0.18)
    band_bottom = bottom - int(height * 0.16)
    if band_bottom <= band_top:
        band_top, band_bottom = top, bottom
    band_height = max(band_bottom - band_top, 1)
    search_start = left + int(width * 0.04)
    search_end = left + int(width * 0.34)
    min_gap = max(18, int(width * 0.04))
    gap_start: Optional[int] = None
    for x in range(search_start, min(search_end, right)):
        dark = 0
        for y in range(max(band_top, 0), min(band_bottom, len(mask))):
            if mask[y][x]:
                dark += 1
        column_density = dark / float(band_height)
        if column_density <= 0.008:
            if gap_start is None:
                gap_start = x
            if x - gap_start + 1 >= min_gap:
                candidate_left = min(right - MIN_CROP_WIDTH, x + 8)
                if candidate_left > left + 20:
                    before_dark = _density(mask, left, top, max(left + 1, gap_start), bottom)
                    after_dark = _density(mask, candidate_left, top, right, bottom)
                    if before_dark > 0 and after_dark >= 0.006:
                        return candidate_left
        else:
            gap_start = None
    return None


def _row_gap_trim_bottom(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> Optional[int]:
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    search_start = top + int(height * 0.54)
    search_end = top + int(height * 0.88)
    min_gap = max(12, int(height * 0.014))
    gap_start: Optional[int] = None
    for y in range(max(search_start, top), min(search_end, bottom)):
        row = mask[y]
        dark = sum(1 for x in range(max(left, 0), min(right, len(row))) if row[x])
        row_density = dark / float(width)
        if row_density <= 0.006:
            if gap_start is None:
                gap_start = y
            if y - gap_start + 1 >= min_gap:
                candidate_bottom = max(top + MIN_CROP_HEIGHT, gap_start + 6)
                if candidate_bottom < bottom - 30:
                    lower_ratio = _row_text_like_ratio(mask, left, candidate_bottom, right, bottom)
                    upper_density = _density(mask, left, top, right, candidate_bottom)
                    if lower_ratio >= 0.05 and upper_density >= 0.006:
                        return candidate_bottom
        else:
            gap_start = None
    return None


def _row_gap_trim_top(mask: Sequence[Sequence[bool]], left: int, top: int, right: int, bottom: int) -> Optional[int]:
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    search_start = top + int(height * 0.04)
    search_end = top + int(height * 0.24)
    min_gap = max(10, int(height * 0.015))
    gap_start: Optional[int] = None
    for y in range(max(search_start, top), min(search_end, bottom)):
        row = mask[y]
        dark = sum(1 for x in range(max(left, 0), min(right, len(row))) if row[x])
        row_density = dark / float(width)
        if row_density <= 0.006:
            if gap_start is None:
                gap_start = y
            if y - gap_start + 1 >= min_gap:
                candidate_top = max(top, gap_start)
                if candidate_top > top + 20 and bottom - candidate_top >= MIN_CROP_HEIGHT:
                    upper_density = _density(mask, left, top, right, candidate_top)
                    lower_density = _density(mask, left, y + 1, right, min(bottom, y + 1 + int(height * 0.24)))
                    if upper_density >= 0.006 and lower_density >= 0.006:
                        return candidate_top
        else:
            gap_start = None
    return None


def _content_bounds(mask: Sequence[Sequence[bool]]) -> tuple[int, int, int, int]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    xs: List[int] = []
    ys: List[int] = []
    for y, row in enumerate(mask):
        for x, dark in enumerate(row):
            if dark:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return 0, 0, width, height
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _trim_visual_crop(image_path: Path, *, trim_right_text: bool = True) -> Path:
    try:
        from PIL import Image
    except ImportError:
        return image_path

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
    except Exception as exc:
        logger.warning("Could not open extracted visual for trimming %s: %s", image_path, exc)
        return image_path

    width, height = image.size
    if width < MIN_CROP_WIDTH or height < MIN_CROP_HEIGHT:
        return image_path
    aspect_ratio = width / max(height, 1)
    if aspect_ratio > MAX_CROP_ASPECT_RATIO:
        logger.info("Skipping extreme-aspect visual crop %s (%sx%s)", image_path, width, height)
        return image_path

    raw_path = image_path.with_name(f"{image_path.stem}.raw{image_path.suffix}")
    try:
        if not raw_path.exists():
            shutil.copy2(image_path, raw_path)
    except Exception:
        pass

    mask = _dark_pixel_mask(image)
    left, top, right, bottom = _content_bounds(mask)
    pad_x = max(int(width * 0.03), 6)
    pad_y = max(int(height * 0.025), 4)
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(height, bottom + pad_y)

    if trim_right_text:
        gap_right = _column_gap_trim_right(mask, left, top, right, bottom) if (right - left) / max(bottom - top, 1) > 1.6 else None
        if gap_right is not None:
            right = gap_right

    if trim_right_text and _right_strip_is_text_heavy(mask, left, top, right, bottom):
        gap_right = _column_gap_trim_right(mask, left, top, right, bottom)
        if gap_right is not None:
            right = gap_right

    if trim_right_text and (bottom - top) / max(right - left, 1) > 1.35:
        gap_bottom = _row_gap_trim_bottom(mask, left, top, right, bottom)
        if gap_bottom is not None:
            bottom = gap_bottom
        gap_top = _row_gap_trim_top(mask, left, top, right, bottom)
        if gap_top is not None:
            top = gap_top

    if trim_right_text and (right - left) / max(bottom - top, 1) < 0.9:
        gap_left = _column_gap_trim_left(mask, left, top, right, bottom)
        if gap_left is not None:
            left = gap_left

    min_width = max(MIN_CROP_WIDTH, int(width * 0.45))
    while trim_right_text and right - left > min_width and _right_strip_is_text_heavy(mask, left, top, right, bottom):
        right -= max(8, int((right - left) * 0.04))

    if right <= left or bottom <= top:
        return image_path
    cropped = image.crop((left, top, right, bottom))
    cropped_width, cropped_height = cropped.size
    if cropped_width < MIN_CROP_WIDTH or cropped_height < MIN_CROP_HEIGHT:
        return image_path
    cropped.save(image_path)
    log_event(
        logger,
        logging.INFO,
        "pdf_visual_crop_trimmed",
        image_path=str(image_path),
        raw_image_path=str(raw_path),
        original_width=width,
        original_height=height,
        final_width=cropped_width,
        final_height=cropped_height,
        crop_box=[left, top, right, bottom],
    )
    return image_path


def _vertical_line_score(mask: Sequence[Sequence[bool]]) -> float:
    height = len(mask)
    width = len(mask[0]) if height else 0
    if not width or not height:
        return 0.0
    best = 0.0
    for x in range(0, width, max(1, width // 120)):
        run = 0
        longest = 0
        for y in range(height):
            if mask[y][x]:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        best = max(best, longest / max(height, 1))
    return best


def _horizontal_line_score(mask: Sequence[Sequence[bool]]) -> float:
    height = len(mask)
    width = len(mask[0]) if height else 0
    if not width or not height:
        return 0.0
    best = 0.0
    for y in range(0, height, max(1, height // 120)):
        run = 0
        longest = 0
        row = mask[y]
        for x in range(width):
            if row[x]:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        best = max(best, longest / max(width, 1))
    return best


def _connected_component_count(mask: Sequence[Sequence[bool]], *, stride: int = 3) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    if not width or not height:
        return 0
    sampled_w = max(1, width // stride)
    sampled_h = max(1, height // stride)
    sampled = [
        [mask[min(y * stride, height - 1)][min(x * stride, width - 1)] for x in range(sampled_w)]
        for y in range(sampled_h)
    ]
    seen = set()
    components = 0
    for y in range(sampled_h):
        for x in range(sampled_w):
            if not sampled[y][x] or (x, y) in seen:
                continue
            components += 1
            stack = [(x, y)]
            seen.add((x, y))
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < sampled_w and 0 <= ny < sampled_h and sampled[ny][nx] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
    return components


def _quality_score_for_crop(image_path: Path, visual_type: str) -> Dict[str, object]:
    try:
        from PIL import Image
    except ImportError:
        return {"score": 0.65, "usable": True, "reasons": ["pil_unavailable"], "metrics": {}}

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
    except Exception as exc:
        return {"score": 0.0, "usable": False, "reasons": [f"open_failed:{exc}"], "metrics": {}}

    width, height = image.size
    reasons: List[str] = []
    if width < MIN_CROP_WIDTH:
        reasons.append("too_narrow")
    if height < MIN_CROP_HEIGHT:
        reasons.append("too_short")
    aspect_ratio = width / max(height, 1)
    if aspect_ratio > MAX_CROP_ASPECT_RATIO:
        reasons.append("extreme_aspect_ratio")

    mask = _dark_pixel_mask(image)
    total_density = _density(mask, 0, 0, width, height)
    top_density = _density(mask, 0, 0, width, int(height * 0.20))
    body_density = _density(mask, 0, int(height * 0.20), width, int(height * 0.82))
    bottom_density = _density(mask, 0, int(height * 0.70), width, height)
    row_ratio = _row_text_like_ratio(mask, 0, 0, width, height)
    right_heavy = _right_strip_is_text_heavy(mask, 0, 0, width, height)
    vertical_score = _vertical_line_score(mask)
    horizontal_score = _horizontal_line_score(mask)
    component_count = _connected_component_count(mask)
    content_left, content_top, content_right, content_bottom = _content_bounds(mask)
    content_area_ratio = ((content_right - content_left) * (content_bottom - content_top)) / max(width * height, 1)

    score = 0.0
    if visual_type == "chart":
        has_axis = vertical_score >= 0.18 or horizontal_score >= 0.18
        has_marks = component_count >= 8 or body_density >= 0.012
        has_body = content_area_ratio >= 0.18 and body_density >= 0.004
        has_bottom_context = bottom_density >= 0.002 or horizontal_score >= 0.18
        if has_axis:
            score += 0.25
        else:
            reasons.append("missing_axis_line")
        if has_marks:
            score += 0.25
        else:
            reasons.append("missing_plotted_marks")
        if has_body:
            score += 0.25
        else:
            reasons.append("weak_chart_body")
        if has_bottom_context:
            score += 0.15
        else:
            reasons.append("missing_x_axis_or_legend")
        if right_heavy:
            score -= 0.18
            reasons.append("right_edge_text_heavy")
    elif visual_type == "table":
        has_grid = vertical_score >= 0.12 and horizontal_score >= 0.12
        has_rows = component_count >= 12 or row_ratio >= 0.10
        has_header = top_density >= 0.008
        if has_grid:
            score += 0.34
        else:
            reasons.append("missing_grid_structure")
        if has_rows:
            score += 0.28
        else:
            reasons.append("missing_data_rows")
        if has_header:
            score += 0.22
        else:
            reasons.append("missing_header_row")
        if right_heavy and aspect_ratio > 2.2:
            score -= 0.10
            reasons.append("possible_adjacent_text")
    else:
        has_shapes = component_count >= 5 or total_density >= 0.015
        has_visual_area = content_area_ratio >= 0.14
        paragraph_like = row_ratio >= 0.45 and vertical_score < 0.12 and horizontal_score < 0.12
        if has_shapes:
            score += 0.35
        else:
            reasons.append("missing_shapes_or_marks")
        if has_visual_area:
            score += 0.30
        else:
            reasons.append("weak_visual_area")
        if paragraph_like:
            score -= 0.25
            reasons.append("mostly_paragraph_text")

    if total_density < 0.002:
        score -= 0.25
        reasons.append("mostly_blank")
    score = max(0.0, min(1.0, score))
    usable = score >= MIN_CROP_QUALITY_SCORE and not any(reason in reasons for reason in ("too_narrow", "too_short", "extreme_aspect_ratio"))
    return {
        "score": round(score, 3),
        "usable": usable,
        "reasons": reasons,
        "metrics": {
            "width": width,
            "height": height,
            "density": round(total_density, 4),
            "body_density": round(body_density, 4),
            "row_text_like_ratio": round(row_ratio, 4),
            "vertical_line_score": round(vertical_score, 4),
            "horizontal_line_score": round(horizontal_score, 4),
            "component_count": component_count,
            "content_area_ratio": round(content_area_ratio, 4),
            "right_edge_text_heavy": right_heavy,
        },
    }


def _visual_type(element: LayoutElement) -> str:
    joined = f"{element.category} {element.text}".lower()
    if "table" in joined:
        return "table"
    for visual_type, keywords in VISUAL_TYPE_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            return visual_type
    return "figure"


def _topic_keywords(text: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "this",
        "that",
        "figure",
        "table",
        "chart",
        "panel",
        "image",
        "source",
        "note",
        "notes",
    }
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= 10:
            break
    return ", ".join(keywords)


def _nearby_text(elements: Sequence[LayoutElement], visual_index: int, page_number: Optional[int]) -> str:
    snippets: List[str] = []
    for index in range(max(0, visual_index - 3), min(len(elements), visual_index + 4)):
        if index == visual_index:
            continue
        element = elements[index]
        if page_number is not None and element.page_number not in (None, page_number):
            continue
        if element.category not in TEXT_CATEGORIES and element.category not in {"FigureCaption"}:
            continue
        text = _full_sentences(element.text, max_sentences=1)
        if text and not BOILERPLATE_PATTERN.search(text):
            snippets.append(text)
    return " ".join(snippets[:3])


def _neighbor_text(
    elements: Sequence[LayoutElement],
    visual_index: int,
    *,
    direction: int,
    page_number: Optional[int],
) -> str:
    index = visual_index + direction
    while 0 <= index < len(elements):
        element = elements[index]
        if page_number is not None and element.page_number not in (None, page_number):
            index += direction
            continue
        if element.category in TEXT_CATEGORIES:
            text = _full_sentences(element.text, max_sentences=2)
            if text and not BOILERPLATE_PATTERN.search(text):
                return text
        index += direction
    return ""


def _combo_chunk(
    *,
    previous_text: str,
    visual_data: str,
    next_text: str,
) -> str:
    return (
        f"[CONTEXT BEFORE]: {_clean_text(previous_text)} | "
        f"[VISUAL DATA]: {_clean_text(visual_data)} | "
        f"[CONTEXT AFTER]: {_clean_text(next_text)}"
    )


def _caption_for_visual(elements: Sequence[LayoutElement], visual_index: int, page_number: Optional[int]) -> str:
    visual = elements[visual_index]
    if visual.text and CAPTION_PATTERN.search(visual.text):
        return _clean_caption_text(visual.text)
    if visual.category in {"Table", "FigureCaption"} and visual.text:
        return _clean_caption_text(visual.text)

    candidates: List[tuple[int, str]] = []
    for index in range(max(0, visual_index - 4), min(len(elements), visual_index + 5)):
        element = elements[index]
        if page_number is not None and element.page_number not in (None, page_number):
            continue
        if element.category not in TEXT_CATEGORIES and element.category != "FigureCaption":
            continue
        text = _clean_text(element.text)
        if not text or BOILERPLATE_PATTERN.search(text):
            continue
        if CAPTION_PATTERN.search(text) or element.category == "FigureCaption":
            candidates.append((abs(index - visual_index), text))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return _clean_caption_text(candidates[0][1])

    fallback = _full_sentences(visual.text, max_sentences=2)
    return _clean_caption_text(fallback) if fallback and not BOILERPLATE_PATTERN.search(fallback) else ""


def _infer_caption_from_text(text: str) -> str:
    cleaned = _clean_text(text)
    match = CAPTION_TEXT_PATTERN.search(cleaned)
    if not match:
        return ""
    caption = _clean_caption_text(match.group(1))
    caption = re.split(r"\s+\[(?:CONTEXT|VISUAL)\s+", caption, maxsplit=1)[0]
    caption = caption.strip(" |")
    return caption


def _is_low_value_visual(element: LayoutElement, caption: str, image_path: str) -> bool:
    if BOILERPLATE_PATTERN.search(caption):
        return True
    if element.category == "Image" and not image_path:
        return True
    if not caption and element.category != "Table":
        return True
    if caption and len(re.findall(r"[A-Za-z]{3,}", caption)) < 3 and element.category != "Table":
        return True
    return False


def _table_markdown(element: LayoutElement) -> str:
    if element.html:
        return element.html
    return element.text


def _caption_image_with_gemini(image_path: str, caption: str = "", nearby_text: str = "") -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not image_path or not Path(image_path).exists():
        log_event(
            logger,
            logging.INFO,
            "vision_captioning_fallback_used",
            reason="missing_api_key_or_image",
            image_path=image_path,
            vision_captioning_enabled=False,
        )
        return ""

    try:
        from google import genai
        from google.genai import types
        from PIL import Image
    except ImportError as exc:
        logger.warning("Gemini captioning skipped because google-genai or Pillow is not installed: %s", exc)
        log_event(
            logger,
            logging.WARNING,
            "vision_captioning_fallback_used",
            reason="google_genai_not_installed",
            image_path=image_path,
            vision_captioning_enabled=False,
        )
        return ""

    try:
        print("--- Sending Image to Gemini for Captioning... ---", flush=True)
        log_event(
            logger,
            logging.INFO,
            "vision_captioning_enabled",
            image_path=image_path,
            model=GEMINI_MODEL_NAME,
            element_type="image",
        )
        client = genai.Client(api_key=api_key)
        img = Image.open(image_path)
        
        user_prompt = "Analyze the provided image."
        if caption:
            user_prompt += f"\n\nFigure/Image Caption from PDF: {caption}"
        if nearby_text:
            user_prompt += f"\n\nNearby text context from the page: {nearby_text}"

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[img, user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        caption_res = _clean_text(response.text or "")
        if caption_res:
            log_event(
                logger,
                logging.INFO,
                "vision_captioning_success",
                image_path=image_path,
                model=GEMINI_MODEL_NAME,
                caption_length=len(caption_res),
            )
            return caption_res
        log_event(
            logger,
            logging.WARNING,
            "vision_captioning_fallback_used",
            reason="empty_vision_caption",
            image_path=image_path,
            vision_captioning_enabled=True,
        )
        return ""
    except Exception as exc:
        logger.warning("Gemini captioning failed for %s: %s", image_path, exc)
        log_event(
            logger,
            logging.WARNING,
            "vision_captioning_failed",
            image_path=image_path,
            model=GEMINI_MODEL_NAME,
            reason=str(exc),
        )
        log_event(
            logger,
            logging.INFO,
            "vision_captioning_fallback_used",
            reason="vision_captioning_failed",
            image_path=image_path,
            vision_captioning_enabled=True,
        )
        return ""


def _vision_caption_status(gemini_caption: str) -> str:
    return "success" if _clean_text(gemini_caption) else "fallback"


def _caption_table_with_gemini(table_text: str, caption: str = "", nearby_text: str = "") -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not table_text:
        log_event(
            logger,
            logging.INFO,
            "vision_captioning_fallback_used",
            reason="missing_api_key_or_table_text",
            element_type="table",
            vision_captioning_enabled=False,
        )
        return ""

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        logger.warning("Gemini table captioning skipped because google-genai is not installed: %s", exc)
        log_event(
            logger,
            logging.WARNING,
            "vision_captioning_fallback_used",
            reason="google_genai_not_installed",
            element_type="table",
            vision_captioning_enabled=False,
        )
        return ""

    try:
        log_event(
            logger,
            logging.INFO,
            "vision_captioning_enabled",
            model=GEMINI_MODEL_NAME,
            element_type="table",
        )
        client = genai.Client(api_key=api_key)
        
        user_prompt = f"Analyze the following table data:\n\n{table_text}"
        if caption:
            user_prompt += f"\n\nTable Caption from PDF: {caption}"
        if nearby_text:
            user_prompt += f"\n\nNearby text context from the page: {nearby_text}"

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        caption_res = _clean_text(response.text or "")
        if caption_res:
            log_event(
                logger,
                logging.INFO,
                "vision_captioning_success",
                model=GEMINI_MODEL_NAME,
                element_type="table",
                caption_length=len(caption_res),
            )
            return caption_res
        log_event(
            logger,
            logging.WARNING,
            "vision_captioning_fallback_used",
            reason="empty_vision_caption",
            element_type="table",
            vision_captioning_enabled=True,
        )
        return ""
    except Exception as exc:
        logger.warning("Gemini table captioning failed: %s", exc)
        log_event(
            logger,
            logging.WARNING,
            "vision_captioning_failed",
            model=GEMINI_MODEL_NAME,
            element_type="table",
            reason=str(exc),
        )
        log_event(
            logger,
            logging.INFO,
            "vision_captioning_fallback_used",
            reason="vision_captioning_failed",
            element_type="table",
            vision_captioning_enabled=True,
        )
        return ""


def _generated_description(
    element: LayoutElement,
    *,
    image_path: str,
    caption: str,
    nearby_text: str,
) -> str:
    generated_description, _status = _generated_description_with_status(
        element,
        image_path=image_path,
        caption=caption,
        nearby_text=nearby_text,
    )
    return generated_description


def _fallback_generated_description(element: LayoutElement, *, caption: str, nearby_text: str) -> str:
    parts = [
        caption,
        _table_markdown(element) if element.category == "Table" else "",
        nearby_text,
    ]
    return _full_sentences(" ".join(part for part in parts if part), max_sentences=5)


def _generated_description_with_status(
    element: LayoutElement,
    *,
    image_path: str,
    caption: str,
    nearby_text: str,
) -> tuple[str, str]:
    if element.category == "Table":
        gemini_caption = _caption_table_with_gemini(_table_markdown(element), caption=caption, nearby_text=nearby_text)
    else:
        gemini_caption = _caption_image_with_gemini(image_path, caption=caption, nearby_text=nearby_text)

    if gemini_caption:
        return gemini_caption, "success"

    return _fallback_generated_description(element, caption=caption, nearby_text=nearby_text), "fallback"


def _visual_data_for_combo(element: LayoutElement, generated_description: str, caption: str) -> str:
    if element.category == "Table":
        table_text = _table_markdown(element)
        parts = [caption, generated_description, table_text]
        return "\n\n".join(part for part in parts if _clean_text(part))
    clean_description = generated_description
    if caption and generated_description:
        clean_description = re.sub(re.escape(caption), "", generated_description, flags=re.IGNORECASE).strip(" .")
    return " ".join(part for part in (caption, clean_description) if _clean_text(part)).strip()


def _visual_document(
    *,
    pdf_path: Path,
    element: LayoutElement,
    element_type: str,
    visual_type: str,
    image_path: str,
    caption: str,
    figure_id: str,
    section: str,
    nearby_text: str,
    previous_text: str,
    next_text: str,
    generated_description: str,
    vision_captioning_status: str,
) -> Document:
    if not image_path and visual_type != "table":
        raise ValueError(
            f"Refusing to create visual document without image_path: "
            f"pdf={pdf_path} page={element.page_number} visual_type={visual_type}"
        )
    if image_path and not Path(image_path).exists():
        raise FileNotFoundError(
            f"Refusing to create visual document because image_path does not exist: {image_path}"
        )
    page_number = element.page_number
    resolved_caption = caption or _infer_caption_from_text(
        " ".join([element.text, generated_description, nearby_text, previous_text, next_text])
    )
    resolved_caption = _clean_caption_text(resolved_caption)
    visual_data = _visual_data_for_combo(element, generated_description, resolved_caption)
    combo_content = _combo_chunk(
        previous_text=previous_text,
        visual_data=visual_data,
        next_text=next_text,
    )
    keywords = _topic_keywords(" ".join([resolved_caption, generated_description, nearby_text, previous_text, next_text]))
    source_page = str(page_number or "")
    resolved_figure_id = figure_id or _figure_id_from_caption(resolved_caption, visual_type, page_number)
    resolved_section = section or _section_from_figure_id(resolved_figure_id) or _topic_keywords(resolved_caption)
    return Document(
        page_content=combo_content,
        metadata={
            "source": str(pdf_path),
            "source_files": pdf_path.name,
            "source_pdf": pdf_path.name,
            "source_type": "pdf",
            "dataset_type": "pdf",
            "content_type": "visual",
            "element_type": element_type,
            "visual_type": visual_type,
            "contains_chart": visual_type != "table",
            "contains_table": visual_type == "table",
            "figure_id": resolved_figure_id,
            "section": resolved_section,
            "section_header": resolved_section,
            "topic": keywords,
            "page": page_number or "",
            "page_number": page_number or "",
            "source_page": page_number or "",
            "image_path": image_path,
            "image_local_path": image_path,
            "raw_image_path": element.raw_image_path or (str(Path(image_path).with_name(f"{Path(image_path).stem}.raw{Path(image_path).suffix}")) if image_path else ""),
            "final_image_path": element.final_image_path or image_path,
            "crop_quality": element.crop_quality,
            "crop_quality_score": element.crop_quality_score,
            "crop_rejected_reason": element.rejected_reason,
            "extraction_pass": element.extraction_pass,
            "raw_box": element.raw_box,
            "final_box": element.final_box,
            "is_multimodal": True,
            "caption": resolved_caption,
            "previous_text": previous_text,
            "next_text": next_text,
            "visual_data": visual_data,
            "nearby_text": nearby_text,
            "generated_description": generated_description,
            "vision_captioning_status": vision_captioning_status,
            "caption_source": "gemini" if vision_captioning_status == "success" else "deterministic_fallback",
            "extraction_method": element.extraction_method or "unstructured_hi_res",
        },
    )


def _visual_elements(elements: Sequence[LayoutElement]) -> Iterable[tuple[int, LayoutElement]]:
    for index, element in enumerate(elements):
        if element.category in {"Image", "Table"}:
            yield index, element


def extract_pdf_visual_documents(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    output_dir: Path = DEFAULT_VISUAL_DIR,
    *,
    max_pages: Optional[int] = None,
) -> List[Document]:
    if not pdf_dir.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Extracted images folder: {output_dir.resolve()} ---", flush=True)
    print(f"--- Visual extraction page range: {START_PAGE}-{END_PAGE} ---", flush=True)
    captioning_enabled = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    log_event(
        logger,
        logging.INFO,
        "gemini_model_configured",
        model=GEMINI_MODEL_NAME,
        env_var="GEMINI_VISION_MODEL",
        fallback_env_var="GEMINI_CAPTION_MODEL",
        vision_captioning_enabled=captioning_enabled,
    )
    log_event(
        logger,
        logging.INFO,
        "vision_captioning_enabled",
        enabled=captioning_enabled,
        model=GEMINI_MODEL_NAME,
        stage="pdf_visual_extraction_startup",
    )
    documents: List[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        elements = _partition_pdf(pdf_path, output_dir, start_page=START_PAGE, end_page=END_PAGE)
        if max_pages is not None:
            elements = [
                element
                for element in elements
                if element.page_number is None or int(element.page_number) <= max_pages
            ]

        extracted_count = 0
        skipped_count = 0
        per_page_counts: Dict[int, int] = defaultdict(int)
        seen_caption_keys = set()
        for visual_index, element in _visual_elements(elements):
            page_number = element.page_number
            caption = _caption_for_visual(elements, visual_index, page_number)
            caption_key = (page_number, _caption_similarity_key(caption))
            if caption_key[1] and caption_key in seen_caption_keys:
                skipped_count += 1
                continue
            if caption_key[1]:
                seen_caption_keys.add(caption_key)
            element_type = "table" if element.category == "Table" else "image"
            visual_type = "table" if element_type == "table" else _visual_type_from_caption(caption)
            if visual_type == "figure":
                visual_type = _visual_type(element)
            page_key = int(page_number or 0)
            per_page_counts[page_key] += 1
            if element.extraction_method == "pymupdf_page_crop":
                image_path = element.image_path
            else:
                image_path = _normalize_image_path(
                    element.image_path,
                    output_dir,
                    pdf_path,
                    per_page_counts[page_key],
                    page_number=page_number,
                    visual_type=visual_type,
                )
                crop_quality_result = _quality_score_for_crop(Path(image_path), visual_type) if image_path else {
                    "score": 0.0,
                    "usable": False,
                    "reasons": ["missing_image_path"],
                    "metrics": {},
                }
                crop_quality = "unstructured_region_accepted" if crop_quality_result.get("usable") else "unstructured_region_low_quality"
                raw_image_path = str(Path(image_path).with_name(f"{Path(image_path).stem}.raw{Path(image_path).suffix}")) if image_path else ""
                final_box = ""
                element = LayoutElement(
                    text=element.text,
                    category=element.category,
                    page_number=element.page_number,
                    image_path=element.image_path,
                    html=element.html,
                    crop_quality=crop_quality,
                    extraction_pass="pass_1",
                    raw_image_path=raw_image_path,
                    final_image_path=image_path,
                    extraction_method="unstructured_hi_res",
                    raw_box="",
                    final_box=final_box,
                    crop_quality_score=float(crop_quality_result.get("score") or 0.0),
                    rejected_reason="" if crop_quality_result.get("usable") else ";".join(str(reason) for reason in crop_quality_result.get("reasons", [])),
                )
                _write_crop_debug(
                    {
                        "source_pdf": str(pdf_path),
                        "page": page_number,
                        "caption": caption,
                        "visual_type": visual_type,
                        "extraction_method": "unstructured_hi_res",
                        "raw_box": [],
                        "final_box": [],
                        "crop_quality_score": element.crop_quality_score,
                        "crop_quality_usable": bool(crop_quality_result.get("usable")),
                        "rejected_reason": element.rejected_reason,
                        "crop_pass_used": element.extraction_pass,
                        "raw_crop_path": raw_image_path,
                        "final_crop_path": image_path,
                        "quality_metrics": crop_quality_result.get("metrics", {}),
                    }
                )
            if image_path:
                print(f"--- Image Detected: {Path(image_path).name} ---", flush=True)
            nearby_text = _nearby_text(elements, visual_index, page_number)
            previous_text = _neighbor_text(elements, visual_index, direction=-1, page_number=page_number)
            next_text = _neighbor_text(elements, visual_index, direction=1, page_number=page_number)
            figure_id = _figure_id_from_caption(caption, visual_type, page_number)
            section = _section_for_visual(elements, visual_index, page_number)
            image_path = _canonicalize_visual_image_path(
                image_path,
                output_dir,
                pdf_path,
                figure_id,
                page_number,
            )
            if image_path:
                element = LayoutElement(
                    text=element.text,
                    category=element.category,
                    page_number=element.page_number,
                    image_path=image_path,
                    html=element.html,
                    crop_quality=element.crop_quality,
                    extraction_pass=element.extraction_pass,
                    raw_image_path=element.raw_image_path,
                    final_image_path=image_path,
                    extraction_method=element.extraction_method,
                    raw_box=element.raw_box,
                    final_box=element.final_box,
                    crop_quality_score=element.crop_quality_score,
                    rejected_reason=element.rejected_reason,
                )

            if _is_low_value_visual(element, caption, image_path):
                skipped_count += 1
                continue

            if image_path and element.crop_quality_score and element.crop_quality_score < MIN_CROP_QUALITY_SCORE * 0.55:
                skipped_count += 1
                log_event(
                    logger,
                    logging.INFO,
                    "pdf_visual_crop_rejected",
                    source_pdf=str(pdf_path),
                    page=page_number,
                    image_path=image_path,
                    visual_type=visual_type,
                    crop_quality_score=element.crop_quality_score,
                    rejected_reason=element.rejected_reason,
                )
                continue

            generated_description, vision_captioning_status = _generated_description_with_status(
                element,
                image_path=image_path,
                caption=caption,
                nearby_text=nearby_text,
            )
            if not generated_description:
                skipped_count += 1
                continue

            documents.append(
                _visual_document(
                    pdf_path=pdf_path,
                    element=element,
                    element_type=element_type,
                    visual_type=visual_type,
                    image_path=image_path,
                    caption=caption,
                    figure_id=figure_id,
                    section=section,
                    nearby_text=nearby_text,
                    previous_text=previous_text,
                    next_text=next_text,
                    generated_description=generated_description,
                    vision_captioning_status=vision_captioning_status,
                )
            )
            extracted_count += 1

        log_event(
            logger,
            logging.INFO,
            "pdf_visual_unstructured_completed",
            source_pdf=str(pdf_path),
            output_dir=str(output_dir),
            visual_documents=extracted_count,
            skipped_visual_elements=skipped_count,
            total_layout_elements=len(elements),
        )

    log_event(
        logger,
        logging.INFO,
        "pdf_visual_extraction_completed",
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        visual_documents=len(documents),
        extraction_method="unstructured_hi_res",
    )
    return documents
