from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.multimodal_assets import validate_asset_path
from ingestion.schemas import ContentBlock, EnrichedDocument, ExtractedImage, VisionDescription


TOKEN_SOFT_LIMIT = 650
FRONT_MATTER_NUMBER = "0"
FRONT_MATTER_TITLE = "Front Matter"
MARKDOWN_HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
CHAPTER_PATTERN = re.compile(
    r"^(?P<label>Chapter|Spotlight|Part)\s+(?P<number>[A-Za-z]?\d+(?:\.\d+)*)"
    r"(?:\s*[:.\-]\s*|\s+)(?P<title>.+)$",
    flags=re.IGNORECASE,
)
SECTION_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+){1,3})\s+(?P<title>[A-Z][^\n]{2,})$"
)
NUMBERED_CHAPTER_PATTERN = re.compile(
    r"^(?P<number>\d+)\s+(?P<title>[A-Z][^\n]{3,})$"
)
VISUAL_LABEL_PATTERN = re.compile(
    r"^(?P<kind>Figure|Fig\.?|Chart|Diagram|Image|Map|Table|Box|Spotlight)\s+"
    r"(?P<identifier>[A-Za-z]?\d+(?:\.\d+)*)"
    r"(?:\s*[:.\-]\s*|\s+)?(?P<title>.*)$",
    flags=re.IGNORECASE,
)
SOURCE_NOTE_PATTERN = re.compile(r"^(Source|Sources|Note|Notes)\s*:\s*(?P<body>.+)$", flags=re.IGNORECASE)


@dataclass(slots=True)
class PdfPageBlock:
    page_no: int
    text: str
    bbox: list[float]
    index: int


@dataclass(slots=True)
class PdfVisualCandidate:
    entity_id: str
    entity_ids: list[str]
    entity_type: str
    visual_title: str
    caption_text: str
    source_note: str
    page_no: int
    chapter_number: str
    chapter_title: str
    section_title: str
    subsection_title: str
    bbox: list[float]
    context_before: str
    context_after: str


@dataclass(slots=True)
class PdfStructure:
    document_title: str
    headings: list[dict[str, Any]]
    text_blocks: list[ContentBlock]
    chapter_blocks: list[ContentBlock]
    outline_block: ContentBlock | None
    visual_candidates: list[PdfVisualCandidate]
    metadata: dict[str, Any]


def _file_hash(path: Path) -> str:
    return hashlib.sha1(str(path.as_posix()).lower().encode("utf-8")).hexdigest()[:12]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-") or "section"


def _stable_suffix(*parts: Any, length: int = 10) -> str:
    joined = "::".join(str(part or "") for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def normalize_visual_entity_id(kind: str, identifier: str) -> str:
    prefix = (
        "Table" if kind.lower() == "table"
        else "Chart" if kind.lower() == "chart"
        else "Diagram" if kind.lower() == "diagram"
        else "Map" if kind.lower() == "map"
        else "Image" if kind.lower() == "image"
        else "Box" if kind.lower() == "box"
        else "Spotlight" if kind.lower() == "spotlight"
        else "Figure"
    )
    cleaned = re.sub(r"\s+", "", str(identifier or "")).upper()
    return f"{prefix}_{cleaned}"


def extract_visual_entities(text: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in VISUAL_LABEL_PATTERN.finditer(str(text or "")):
        kind_raw = match.group("kind").lower().replace("fig.", "figure").replace("fig", "figure")
        kind = "figure" if kind_raw.startswith("figure") else kind_raw
        identifier = re.sub(r"\s+", "", match.group("identifier"))
        entity_id = normalize_visual_entity_id(kind, identifier)
        if entity_id.lower() in seen:
            continue
        seen.add(entity_id.lower())
        entities.append(
            {
                "kind": kind,
                "identifier": identifier,
                "entity_id": entity_id,
                "label": f"{match.group('kind').replace('Fig.', 'Figure')} {identifier}",
                "title": str(match.group("title") or "").strip(),
            }
        )
    return entities


def _leading_visual_entity(text: str) -> dict[str, str] | None:
    clean = _clean_text(text)
    match = VISUAL_LABEL_PATTERN.match(clean)
    if not match:
        return None
    kind_raw = match.group("kind").lower().replace("fig.", "figure").replace("fig", "figure")
    kind = "figure" if kind_raw.startswith("figure") else kind_raw
    identifier = re.sub(r"\s+", "", match.group("identifier"))
    return {
        "kind": kind,
        "identifier": identifier,
        "entity_id": normalize_visual_entity_id(kind, identifier),
        "label": f"{match.group('kind').replace('Fig.', 'Figure')} {identifier}",
        "title": str(match.group("title") or "").strip(),
    }


def _is_visual_heading(text: str) -> bool:
    return _leading_visual_entity(text) is not None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _looks_like_title(text: str) -> bool:
    clean = _clean_text(text)
    if not clean or len(clean) > 140:
        return False
    if VISUAL_LABEL_PATTERN.match(clean) or CHAPTER_PATTERN.match(clean) or SECTION_PATTERN.match(clean):
        return True
    words = clean.split()
    return 2 <= len(words) <= 14 and clean[:1].isupper() and clean.endswith((".", "?", "!")) is False


def _token_len(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def extract_pdf_page_blocks(pdf_path: Path) -> list[PdfPageBlock]:
    import fitz

    blocks: list[PdfPageBlock] = []
    with fitz.open(str(pdf_path)) as document:
        for page_index, page in enumerate(document, start=1):
            page_blocks = sorted(page.get_text("blocks"), key=lambda item: (item[1], item[0]))
            for block_index, block in enumerate(page_blocks, start=1):
                x0, y0, x1, y1, text, *_rest = block
                clean = _clean_text(text)
                if not clean:
                    continue
                blocks.append(
                    PdfPageBlock(
                        page_no=page_index,
                        text=clean,
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        index=block_index,
                    )
                )
    return blocks


def _infer_document_title(pdf_path: Path, markdown: str, page_blocks: list[PdfPageBlock]) -> str:
    for line in str(markdown or "").splitlines():
        match = MARKDOWN_HEADING_PATTERN.match(line.strip())
        if match:
            return _clean_text(match.group("title"))
    for block in page_blocks[:8]:
        if _looks_like_title(block.text):
            return block.text
    return pdf_path.stem.replace("_", " ")


def _major_heading(text: str, block_index: int = 0) -> tuple[str, str, str] | None:
    match = CHAPTER_PATTERN.match(text)
    if match:
        label = match.group("label").strip()
        number = match.group("number").strip()
        title = _clean_text(match.group("title"))
        word_count = len(title.split())
        if not title:
            return None
        if len(title) > 110 or word_count > 16:
            return None
        if title.endswith((".", "?", "!")):
            return None
        return (label, number, title)

    numbered_match = NUMBERED_CHAPTER_PATTERN.match(text)
    if not numbered_match or block_index > 2:
        return None
    number = numbered_match.group("number").strip()
    title = _clean_text(numbered_match.group("title"))
    if len(title) > 110 or len(title.split()) > 16:
        return None
    if title.endswith((".", "?", "!")):
        return None
    return ("Chapter", number, title)


def _section_heading(text: str) -> tuple[str, str] | None:
    match = SECTION_PATTERN.match(text)
    if not match:
        return None
    return match.group("number").strip(), _clean_text(match.group("title"))


def _visual_kind_from_entity(entity_kind: str, title: str, description: str = "") -> str:
    combined = f"{title} {description}".lower()
    if entity_kind == "table":
        return "table"
    if entity_kind == "map":
        return "map"
    if entity_kind == "diagram" or any(token in combined for token in ("workflow", "process", "diagram", "tree", "system")):
        return "diagram"
    if entity_kind == "chart" or any(token in combined for token in ("axis", "axes", "bar", "line", "scatter", "trend", "plot")):
        return "chart"
    if entity_kind == "image":
        return "image"
    return "figure"


def _source_note_lines(texts: list[str]) -> tuple[str, list[str]]:
    note_lines: list[str] = []
    body_lines: list[str] = []
    for text in texts:
        if SOURCE_NOTE_PATTERN.match(text):
            note_lines.append(text)
        else:
            body_lines.append(text)
    return "\n".join(note_lines).strip(), body_lines


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _page_text_lookup(page_blocks: list[PdfPageBlock]) -> dict[int, str]:
    page_texts: dict[int, list[str]] = {}
    for block in page_blocks:
        page_texts.setdefault(block.page_no, []).append(block.text)
    return {page_no: _normalize_for_match(" ".join(parts)) for page_no, parts in page_texts.items()}


def _infer_page_for_text(text: str, page_texts: dict[int, str], last_page: int = 1) -> int:
    if not page_texts:
        return 1
    probe = _normalize_for_match(text)
    if not probe:
        return last_page
    probe = probe[:160].strip()
    ordered_pages = sorted(page_texts)
    forward_pages = [page for page in ordered_pages if page >= last_page]
    search_pages = forward_pages + [page for page in ordered_pages if page < last_page]
    for page_no in search_pages:
        if probe and probe in page_texts[page_no]:
            return page_no
    probe_tokens = [token for token in probe.split() if len(token) >= 5][:10]
    for page_no in search_pages:
        page_text = page_texts[page_no]
        if probe_tokens and sum(1 for token in probe_tokens if token in page_text) >= max(2, min(4, len(probe_tokens))):
            return page_no
    return last_page if last_page in page_texts else ordered_pages[0]


def _iter_markdown_segments(markdown: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = _clean_text(" ".join(paragraph_lines))
        if paragraph:
            segments.append({"kind": "paragraph", "text": paragraph})
        paragraph_lines = []

    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("```"):
            in_code_block = not in_code_block
            flush_paragraph()
            continue
        if in_code_block:
            continue
        heading_match = MARKDOWN_HEADING_PATTERN.match(line)
        if heading_match:
            flush_paragraph()
            segments.append(
                {
                    "kind": "heading",
                    "level": len(heading_match.group("level")),
                    "text": _clean_text(heading_match.group("title")),
                }
            )
            continue
        if line.startswith("|") or re.match(r"^\s*[-:|]+\s*$", line):
            flush_paragraph()
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return segments


def _build_markdown_text_structure(
    pdf_path: Path,
    markdown: str,
    page_blocks: list[PdfPageBlock],
    document_title: str,
    file_hash: str,
) -> tuple[list[dict[str, Any]], list[ContentBlock], list[ContentBlock]]:
    page_texts = _page_text_lookup(page_blocks)
    headings: list[dict[str, Any]] = []
    chapter_blocks: list[ContentBlock] = []
    text_blocks: list[ContentBlock] = []
    chapter_ranges: list[dict[str, Any]] = []

    current_chapter_number = FRONT_MATTER_NUMBER
    current_chapter_title = FRONT_MATTER_TITLE
    current_section_title = FRONT_MATTER_TITLE
    current_subsection_title = ""
    active_texts: list[str] = []
    active_text_meta: dict[str, Any] | None = None
    last_page = 1

    def current_parent_id() -> str:
        return (
            chapter_ranges[-1]["parent_id"]
            if chapter_ranges
            else f"pdf::{file_hash}::chapter::{_slug(FRONT_MATTER_TITLE)}::{_slug(FRONT_MATTER_NUMBER)}"
        )

    def flush_text_block() -> None:
        nonlocal active_texts, active_text_meta
        if not active_texts or not active_text_meta:
            active_texts = []
            active_text_meta = None
            return
        text = "\n\n".join(active_texts).strip()
        if text:
            page_no = active_text_meta["page_no"]
            metadata = {
                "chunk_id": (
                    f"pdf::{file_hash}::section::{active_text_meta['chapter_number']}"
                    f"::{_slug(active_text_meta['section_title'])}::page::{page_no}::text::{len(text_blocks) + 1}"
                ),
                "document_type": "pdf",
                "chunk_type": "section_text_chunk",
                "entity_type": "text",
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
                "document_title": document_title,
                "page_no": page_no,
                "page_start": page_no,
                "page_end": page_no,
                "chapter_number": active_text_meta["chapter_number"],
                "chapter_title": active_text_meta["chapter_title"],
                "section_title": active_text_meta["section_title"],
                "subsection_title": active_text_meta["subsection_title"],
                "parent_id": active_text_meta["parent_id"],
            }
            text_blocks.append(ContentBlock(text=text, type="text", page=page_no, source_path=str(pdf_path), metadata=metadata))
        active_texts = []
        active_text_meta = None

    for segment in _iter_markdown_segments(markdown):
        text = segment["text"]
        if not text:
            continue
        if text == document_title and segment["kind"] != "heading":
            continue
        if _is_visual_heading(text) or SOURCE_NOTE_PATTERN.match(text):
            continue

        if segment["kind"] == "heading":
            page_no = _infer_page_for_text(text, page_texts, last_page)
            last_page = page_no
            major = _major_heading(text, 0)
            section = _section_heading(text)
            if major:
                flush_text_block()
                label, number, title = major
                if chapter_ranges:
                    chapter_ranges[-1]["page_end"] = max(chapter_ranges[-1]["page_start"], page_no - 1)
                current_chapter_number = number
                current_chapter_title = f"{label} {number}: {title}"
                current_section_title = current_chapter_title
                current_subsection_title = ""
                chapter_parent_id = f"pdf::{file_hash}::chapter::{_slug(label)}::{_slug(number)}"
                chapter_ranges.append(
                    {
                        "parent_id": chapter_parent_id,
                        "chapter_number": number,
                        "chapter_title": current_chapter_title,
                        "page_start": page_no,
                        "page_end": page_no,
                    }
                )
                headings.append({"level": 1, "title": current_chapter_title, "page_no": page_no})
                chapter_blocks.append(
                    ContentBlock(
                        text=f"{current_chapter_title}\n\nThis chapter begins on page {page_no}.",
                        type="text",
                        page=page_no,
                        source_path=str(pdf_path),
                        metadata={
                            "chunk_id": chapter_parent_id,
                            "document_type": "pdf",
                            "chunk_type": "chapter_chunk",
                            "entity_type": "chapter",
                            "source_file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "document_title": document_title,
                            "page_no": page_no,
                            "page_start": page_no,
                            "page_end": page_no,
                            "chapter_number": number,
                            "chapter_title": current_chapter_title,
                            "section_title": current_chapter_title,
                            "subsection_title": "",
                            "parent_id": chapter_parent_id,
                        },
                    )
                )
                continue

            flush_text_block()
            if section:
                number, title = section
                if number.count(".") >= 2:
                    current_subsection_title = f"{number} {title}"
                    headings.append({"level": 3, "title": current_subsection_title, "page_no": page_no})
                else:
                    current_section_title = f"{number} {title}"
                    current_subsection_title = ""
                    headings.append({"level": 2, "title": current_section_title, "page_no": page_no})
                continue

            level = int(segment.get("level") or 2)
            if level <= 2:
                current_section_title = text
                current_subsection_title = ""
                headings.append({"level": 2, "title": current_section_title, "page_no": page_no})
            else:
                current_subsection_title = text
                headings.append({"level": 3, "title": current_subsection_title, "page_no": page_no})
            continue

        page_no = _infer_page_for_text(text, page_texts, last_page)
        last_page = page_no
        if not active_text_meta:
            active_text_meta = {
                "page_no": page_no,
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "section_title": current_section_title or current_chapter_title,
                "subsection_title": current_subsection_title,
                "parent_id": current_parent_id(),
            }
        same_group = (
            active_text_meta["page_no"] == page_no
            and active_text_meta["chapter_number"] == current_chapter_number
            and active_text_meta["section_title"] == (current_section_title or current_chapter_title)
            and active_text_meta["subsection_title"] == current_subsection_title
        )
        if not same_group or _token_len("\n\n".join([*active_texts, text])) > TOKEN_SOFT_LIMIT:
            flush_text_block()
            active_text_meta = {
                "page_no": page_no,
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "section_title": current_section_title or current_chapter_title,
                "subsection_title": current_subsection_title,
                "parent_id": current_parent_id(),
            }
        active_texts.append(text)

    flush_text_block()
    if chapter_ranges:
        last_page_no = page_blocks[-1].page_no if page_blocks else 1
        chapter_ranges[-1]["page_end"] = max(chapter_ranges[-1]["page_start"], last_page_no)
        for chapter_block, chapter_range in zip(chapter_blocks, chapter_ranges):
            chapter_block.metadata["page_start"] = chapter_range["page_start"]
            chapter_block.metadata["page_end"] = chapter_range["page_end"]
            chapter_block.text = (
                f"{chapter_range['chapter_title']}\n\n"
                f"This chapter spans pages {chapter_range['page_start']} to {chapter_range['page_end']}."
            )

    return headings, chapter_blocks, text_blocks


def _context_for_page(
    page_no: int,
    chapter_blocks: list[ContentBlock],
    text_blocks: list[ContentBlock],
) -> dict[str, str]:
    chapter_number = FRONT_MATTER_NUMBER
    chapter_title = FRONT_MATTER_TITLE
    section_title = FRONT_MATTER_TITLE
    subsection_title = ""
    for block in chapter_blocks:
        metadata = dict(block.metadata or {})
        start = int(metadata.get("page_start") or metadata.get("page_no") or 0)
        end = int(metadata.get("page_end") or start)
        if start <= page_no <= end:
            chapter_number = str(metadata.get("chapter_number") or FRONT_MATTER_NUMBER)
            chapter_title = str(metadata.get("chapter_title") or FRONT_MATTER_TITLE)
            section_title = str(metadata.get("section_title") or chapter_title)
            subsection_title = str(metadata.get("subsection_title") or "")
            break
    for block in text_blocks:
        metadata = dict(block.metadata or {})
        if int(metadata.get("page_no") or 0) > page_no:
            continue
        if str(metadata.get("chapter_number") or "") != chapter_number:
            continue
        section_title = str(metadata.get("section_title") or section_title)
        subsection_title = str(metadata.get("subsection_title") or subsection_title)
    return {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "section_title": section_title,
        "subsection_title": subsection_title,
    }


def _build_page_block_text_structure(
    pdf_path: Path,
    page_blocks: list[PdfPageBlock],
    document_title: str,
    file_hash: str,
) -> tuple[list[dict[str, Any]], list[ContentBlock], list[ContentBlock]]:
    headings: list[dict[str, Any]] = []
    chapter_blocks: list[ContentBlock] = []
    text_blocks: list[ContentBlock] = []
    chapter_ranges: list[dict[str, Any]] = []

    current_chapter_number = FRONT_MATTER_NUMBER
    current_chapter_title = FRONT_MATTER_TITLE
    current_section_title = FRONT_MATTER_TITLE
    current_subsection_title = ""
    active_texts: list[str] = []
    active_text_meta: dict[str, Any] | None = None

    def flush_text_block() -> None:
        nonlocal active_texts, active_text_meta
        if not active_texts or not active_text_meta:
            active_texts = []
            active_text_meta = None
            return
        text = "\n\n".join(active_texts).strip()
        if text:
            page_no = active_text_meta["page_no"]
            metadata = {
                "chunk_id": f"pdf::{file_hash}::page::{page_no}::text::{len(text_blocks) + 1}",
                "document_type": "pdf",
                "chunk_type": "section_text_chunk",
                "entity_type": "text",
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
                "document_title": document_title,
                "page_no": page_no,
                "page_start": page_no,
                "page_end": page_no,
                "chapter_number": active_text_meta["chapter_number"],
                "chapter_title": active_text_meta["chapter_title"],
                "section_title": active_text_meta["section_title"],
                "subsection_title": active_text_meta["subsection_title"],
                "parent_id": active_text_meta["parent_id"],
            }
            text_blocks.append(ContentBlock(text=text, type="text", page=page_no, source_path=str(pdf_path), metadata=metadata))
        active_texts = []
        active_text_meta = None

    for block in page_blocks:
        text = block.text
        major = _major_heading(text, block.index)
        if major:
            flush_text_block()
            label, number, title = major
            if chapter_ranges:
                chapter_ranges[-1]["page_end"] = max(chapter_ranges[-1]["page_start"], block.page_no - 1)
            current_chapter_number = number
            current_chapter_title = f"{label} {number}: {title}"
            current_section_title = current_chapter_title
            current_subsection_title = ""
            chapter_parent_id = f"pdf::{file_hash}::chapter::{_slug(label)}::{_slug(number)}"
            chapter_ranges.append(
                {
                    "parent_id": chapter_parent_id,
                    "chapter_number": number,
                    "chapter_title": current_chapter_title,
                    "page_start": block.page_no,
                    "page_end": block.page_no,
                }
            )
            headings.append({"level": 1, "title": current_chapter_title, "page_no": block.page_no})
            chapter_blocks.append(
                ContentBlock(
                    text=f"{current_chapter_title}\n\nThis chapter begins on page {block.page_no}.",
                    type="text",
                    page=block.page_no,
                    source_path=str(pdf_path),
                    metadata={
                        "chunk_id": chapter_parent_id,
                        "document_type": "pdf",
                        "chunk_type": "chapter_chunk",
                        "entity_type": "chapter",
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "document_title": document_title,
                        "page_no": block.page_no,
                        "page_start": block.page_no,
                        "page_end": block.page_no,
                        "chapter_number": number,
                        "chapter_title": current_chapter_title,
                        "section_title": current_chapter_title,
                        "subsection_title": "",
                        "parent_id": chapter_parent_id,
                    },
                )
            )
            continue

        section = _section_heading(text)
        if section:
            flush_text_block()
            number, title = section
            if number.count(".") >= 2:
                current_subsection_title = f"{number} {title}"
                headings.append({"level": 3, "title": current_subsection_title, "page_no": block.page_no})
            else:
                current_section_title = f"{number} {title}"
                current_subsection_title = ""
                headings.append({"level": 2, "title": current_section_title, "page_no": block.page_no})
            continue

        if _is_visual_heading(text) or SOURCE_NOTE_PATTERN.match(text):
            continue

        if not active_text_meta:
            active_text_meta = {
                "page_no": block.page_no,
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "section_title": current_section_title or current_chapter_title,
                "subsection_title": current_subsection_title,
                "parent_id": (
                    chapter_ranges[-1]["parent_id"]
                    if chapter_ranges
                    else f"pdf::{file_hash}::chapter::{_slug(FRONT_MATTER_TITLE)}::{_slug(FRONT_MATTER_NUMBER)}"
                ),
            }
        same_group = (
            active_text_meta["page_no"] == block.page_no
            and active_text_meta["chapter_number"] == current_chapter_number
            and active_text_meta["section_title"] == (current_section_title or current_chapter_title)
            and active_text_meta["subsection_title"] == current_subsection_title
        )
        if not same_group or _token_len("\n\n".join([*active_texts, text])) > TOKEN_SOFT_LIMIT:
            flush_text_block()
            active_text_meta = {
                "page_no": block.page_no,
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "section_title": current_section_title or current_chapter_title,
                "subsection_title": current_subsection_title,
                "parent_id": (
                    chapter_ranges[-1]["parent_id"]
                    if chapter_ranges
                    else f"pdf::{file_hash}::chapter::{_slug(FRONT_MATTER_TITLE)}::{_slug(FRONT_MATTER_NUMBER)}"
                ),
            }
        active_texts.append(text)

    flush_text_block()
    if chapter_ranges:
        last_page_no = page_blocks[-1].page_no if page_blocks else 1
        chapter_ranges[-1]["page_end"] = max(chapter_ranges[-1]["page_start"], last_page_no)
        for chapter_block, chapter_range in zip(chapter_blocks, chapter_ranges):
            chapter_block.metadata["page_start"] = chapter_range["page_start"]
            chapter_block.metadata["page_end"] = chapter_range["page_end"]
            chapter_block.text = (
                f"{chapter_range['chapter_title']}\n\n"
                f"This chapter spans pages {chapter_range['page_start']} to {chapter_range['page_end']}."
            )

    return headings, chapter_blocks, text_blocks


def _assign_chapter_child_ids(chapter_blocks: list[ContentBlock], text_blocks: list[ContentBlock]) -> None:
    chapter_children: dict[str, list[str]] = {}
    for text_block in text_blocks:
        parent_id = str(text_block.metadata.get("parent_id") or "")
        chunk_id = str(text_block.metadata.get("chunk_id") or "")
        if parent_id and chunk_id:
            chapter_children.setdefault(parent_id, []).append(chunk_id)
    for chapter_block in chapter_blocks:
        parent_id = str(chapter_block.metadata.get("parent_id") or chapter_block.metadata.get("chunk_id") or "")
        chapter_block.metadata["child_ids"] = chapter_children.get(parent_id, [])


def build_pdf_structure(pdf_path: Path, markdown: str, parser_name: str = "docling") -> PdfStructure:
    page_blocks = extract_pdf_page_blocks(pdf_path)
    document_title = _infer_document_title(pdf_path, markdown, page_blocks)
    file_hash = _file_hash(pdf_path)
    headings, chapter_blocks, text_blocks = _build_markdown_text_structure(
        pdf_path=pdf_path,
        markdown=markdown,
        page_blocks=page_blocks,
        document_title=document_title,
        file_hash=file_hash,
    )
    if not chapter_blocks:
        headings, chapter_blocks, text_blocks = _build_page_block_text_structure(
            pdf_path=pdf_path,
            page_blocks=page_blocks,
            document_title=document_title,
            file_hash=file_hash,
        )
    _assign_chapter_child_ids(chapter_blocks, text_blocks)
    visual_candidates: list[PdfVisualCandidate] = []

    for index, block in enumerate(page_blocks):
        text = block.text
        entity = _leading_visual_entity(text)
        if entity:
            note_lines: list[str] = []
            context_after = ""
            lookahead_index = index + 1
            while lookahead_index < len(page_blocks) and page_blocks[lookahead_index].page_no == block.page_no:
                lookahead_text = page_blocks[lookahead_index].text
                if _is_visual_heading(lookahead_text) or _major_heading(lookahead_text, page_blocks[lookahead_index].index) or _section_heading(lookahead_text):
                    break
                if SOURCE_NOTE_PATTERN.match(lookahead_text):
                    note_lines.append(lookahead_text)
                    lookahead_index += 1
                    continue
                if not context_after:
                    context_after = lookahead_text
                break
            source_note, _unused = _source_note_lines(note_lines)
            context_before = ""
            for back_index in range(index - 1, -1, -1):
                candidate = page_blocks[back_index]
                if candidate.page_no != block.page_no:
                    break
                if _is_visual_heading(candidate.text) or _major_heading(candidate.text, candidate.index) or _section_heading(candidate.text):
                    continue
                context_before = candidate.text
                break
            page_context = _context_for_page(block.page_no, chapter_blocks, text_blocks)
            entity_type = _visual_kind_from_entity(entity["kind"], entity["title"])
            title = text
            visual_candidates.append(
                PdfVisualCandidate(
                    entity_id=entity["entity_id"],
                    entity_ids=[entity["entity_id"], entity["label"]],
                    entity_type=entity_type,
                    visual_title=title,
                    caption_text="\n".join(part for part in [title, *note_lines] if part).strip(),
                    source_note=source_note,
                    page_no=block.page_no,
                    chapter_number=page_context["chapter_number"],
                    chapter_title=page_context["chapter_title"],
                    section_title=page_context["section_title"],
                    subsection_title=page_context["subsection_title"],
                    bbox=block.bbox,
                    context_before=context_before,
                    context_after=context_after,
                )
            )

    outline_lines = [document_title]
    for heading in headings:
        indent = "  " * max(int(heading["level"]) - 1, 0)
        outline_lines.append(f"{indent}- {heading['title']} (page {heading['page_no']})")
    outline_block = ContentBlock(
        text="\n".join(outline_lines).strip(),
        type="text",
        page=1 if page_blocks else None,
        source_path=str(pdf_path),
        metadata={
            "chunk_id": f"pdf::{file_hash}::outline",
            "document_type": "pdf",
            "chunk_type": "document_outline_chunk",
            "entity_type": "document_outline",
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "document_title": document_title,
            "page_no": 1 if page_blocks else None,
            "page_start": 1 if page_blocks else None,
            "page_end": page_blocks[-1].page_no if page_blocks else None,
            "chapter_number": FRONT_MATTER_NUMBER,
            "chapter_title": FRONT_MATTER_TITLE,
            "section_title": FRONT_MATTER_TITLE,
            "subsection_title": "",
        },
    )
    return PdfStructure(
        document_title=document_title,
        headings=headings,
        text_blocks=text_blocks,
        chapter_blocks=chapter_blocks,
        outline_block=outline_block,
        visual_candidates=visual_candidates,
        metadata={
            "source_type": "pdf",
            "parser": parser_name,
            "document_title": document_title,
            "page_block_count": len(page_blocks),
            "page_count": page_blocks[-1].page_no if page_blocks else 0,
            "chapter_count": len(chapter_blocks),
            "visual_candidate_count": len(visual_candidates),
        },
    )


def _visual_path_fields(entity_type: str, image_path: str) -> dict[str, Any]:
    if entity_type == "chart":
        return {"chart_image_path": image_path, "image_path": image_path}
    if entity_type == "diagram":
        return {"diagram_image_path": image_path, "image_path": image_path}
    if entity_type == "map":
        return {"image_path": image_path}
    if entity_type in {"figure", "table"}:
        key = "figure_image_path" if entity_type == "figure" else "table_image_path"
        return {key: image_path, "image_path": image_path}
    return {"image_path": image_path}


def _asset_validation_type(entity_type: str, asset_path: str) -> str:
    suffix = Path(str(asset_path or "")).suffix.lower()
    if entity_type == "table" and suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "table_image"
    if entity_type == "table":
        return "table_csv"
    if entity_type == "chart":
        return "chart_image"
    if entity_type == "diagram":
        return "diagram_image"
    if entity_type == "figure":
        return "figure_image"
    if entity_type == "map":
        return "image"
    return entity_type or "image"


def _classify_visual_candidate(candidate: PdfVisualCandidate, description: str = "") -> str:
    return _visual_kind_from_entity(candidate.entity_type, candidate.visual_title, description)


def _match_visual_candidate(candidate: PdfVisualCandidate, assets: list[ExtractedImage]) -> tuple[ExtractedImage | None, str]:
    exact_matches = [
        asset for asset in assets
        if str(asset.metadata.get("entity_id") or "").lower() == candidate.entity_id.lower()
        or candidate.entity_id.lower() in Path(str(asset.image_path)).stem.lower()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0], "exact_entity_id"
    same_page_type = [
        asset for asset in assets
        if asset.page == candidate.page_no and _visual_kind_from_entity(asset.type, str(asset.metadata.get("source_label") or "")) == candidate.entity_type
    ]
    if len(same_page_type) == 1:
        return same_page_type[0], "same_page_same_type"
    same_page = [asset for asset in assets if asset.page == candidate.page_no]
    if len(same_page) == 1:
        return same_page[0], "nearest_same_page"
    return None, "unmatched"


def _page_chapter_context(document: EnrichedDocument, page_no: int | None) -> dict[str, str]:
    if page_no is None:
        return {
            "chapter_number": FRONT_MATTER_NUMBER,
            "chapter_title": FRONT_MATTER_TITLE,
            "section_title": FRONT_MATTER_TITLE,
            "subsection_title": "",
        }
    for block in document.blocks:
        metadata = dict(block.metadata or {})
        if metadata.get("chunk_type") != "chapter_chunk":
            continue
        page_start = int(metadata.get("page_start") or metadata.get("page_no") or 0)
        page_end = int(metadata.get("page_end") or page_start)
        if page_start <= page_no <= page_end:
            return {
                "chapter_number": str(metadata.get("chapter_number") or FRONT_MATTER_NUMBER),
                "chapter_title": str(metadata.get("chapter_title") or FRONT_MATTER_TITLE),
                "section_title": str(metadata.get("section_title") or metadata.get("chapter_title") or FRONT_MATTER_TITLE),
                "subsection_title": str(metadata.get("subsection_title") or ""),
            }
    return {
        "chapter_number": FRONT_MATTER_NUMBER,
        "chapter_title": FRONT_MATTER_TITLE,
        "section_title": FRONT_MATTER_TITLE,
        "subsection_title": "",
    }


def build_visual_blocks(
    pdf_path: Path,
    document: EnrichedDocument,
    visual_candidates: list[PdfVisualCandidate],
    images: list[ExtractedImage],
    descriptions: list[VisionDescription],
) -> list[ContentBlock]:
    file_hash = _file_hash(pdf_path)
    descriptions_by_path = {str(description.image_path): description for description in descriptions}
    blocks: list[ContentBlock] = []
    used_assets: set[str] = set()
    covered_page_types = {(candidate.page_no, candidate.entity_type) for candidate in visual_candidates}

    for candidate in visual_candidates:
        matched_image, match_reason = _match_visual_candidate(candidate, images)
        image_path = ""
        asset_exists = False
        asset_validation_status = "missing"
        asset_validation_reason = "asset path was not matched"
        description_text = ""
        bbox = candidate.bbox
        docling_label = ""
        if matched_image is not None:
            used_assets.add(str(matched_image.image_path))
            image_path = str(Path(matched_image.image_path).resolve())
            validation = validate_asset_path(image_path, _asset_validation_type(candidate.entity_type, image_path))
            asset_exists = validation.ok
            asset_validation_status = "allowed" if validation.ok else "blocked"
            asset_validation_reason = validation.reason
            bbox = matched_image.coordinates.get("bbox") if isinstance(matched_image.coordinates, dict) and matched_image.coordinates.get("bbox") else bbox
            docling_label = str(matched_image.metadata.get("category") or "")
            description = descriptions_by_path.get(str(matched_image.image_path))
            if description is not None:
                description_text = description.description
        elif candidate.page_no is not None:
            page_assets = [
                asset for asset in images
                if asset.page == candidate.page_no and str(asset.image_path) not in used_assets
            ]
            if len(page_assets) == 1:
                matched_image = page_assets[0]
                used_assets.add(str(matched_image.image_path))
                image_path = str(Path(matched_image.image_path).resolve())
                validation = validate_asset_path(image_path, _asset_validation_type(candidate.entity_type, image_path))
                asset_exists = validation.ok
                asset_validation_status = "allowed" if validation.ok else "blocked"
                asset_validation_reason = validation.reason
                description = descriptions_by_path.get(str(matched_image.image_path))
                if description is not None:
                    description_text = description.description
                match_reason = "page_level_fallback"

        entity_type = _classify_visual_candidate(candidate, description_text)
        visual_parent_id = (
            f"pdf::{file_hash}::visual::{candidate.entity_id}"
            f"::page::{candidate.page_no}"
            f"::{_stable_suffix(candidate.caption_text, candidate.context_before, candidate.context_after)}"
        )
        flags = {
            "contains_figure": entity_type in {"figure", "chart", "diagram"},
            "contains_chart": entity_type == "chart",
            "contains_diagram": entity_type == "diagram",
            "contains_image": entity_type in {"figure", "chart", "diagram", "image", "map"},
            "contains_map": entity_type == "map",
            "contains_table": entity_type == "table",
        }
        base_metadata = {
            "document_type": "pdf",
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "document_title": document.metadata.get("document_title", pdf_path.stem),
            "entity_type": entity_type,
            "entity_id": candidate.entity_id,
            "entity_ids": candidate.entity_ids,
            "chapter_number": candidate.chapter_number,
            "chapter_title": candidate.chapter_title,
            "section_title": candidate.section_title,
            "subsection_title": candidate.subsection_title,
            "page_no": candidate.page_no,
            "page_start": candidate.page_no,
            "page_end": candidate.page_no,
            "visual_title": candidate.visual_title,
            "caption_text": candidate.caption_text,
            "source_note": candidate.source_note,
            "bbox": bbox,
            "docling_label": docling_label,
            "docling_self_ref": match_reason,
            "asset_exists": asset_exists,
            "asset_validation_status": asset_validation_status,
            "asset_validation_reason": asset_validation_reason,
            "preserve_child_text": True,
            **flags,
        }

        visual_child_ids = [f"{visual_parent_id}::caption"]

        caption_metadata = {
            **base_metadata,
            "chunk_id": f"{visual_parent_id}::caption",
            "chunk_type": "visual_caption_chunk",
            "parent_id": visual_parent_id,
            "linked_entity_id": candidate.entity_id,
            "linked_entity_type": entity_type,
        }
        if image_path:
            caption_metadata.update(_visual_path_fields(entity_type, image_path))
            caption_metadata["asset_paths"] = [image_path]
            caption_metadata["asset_types"] = [entity_type]

        blocks.append(
            ContentBlock(
                text=candidate.caption_text,
                type="text",
                page=candidate.page_no,
                source_path=str(pdf_path),
                metadata=caption_metadata,
            )
        )

        if image_path:
            asset_fields = _visual_path_fields(entity_type, image_path)
            visual_child_ids.append(f"{visual_parent_id}::asset")
            blocks.append(
                ContentBlock(
                    text=(
                        f"Verified visual asset for {candidate.entity_id}: {candidate.visual_title}.\n"
                        f"{description_text.strip() or 'No generated visual description available.'}"
                    ),
                    type=entity_type if entity_type in {"chart", "diagram", "figure", "image", "map", "table"} else "image",
                    page=candidate.page_no,
                    source_path=str(pdf_path),
                    metadata={
                        **base_metadata,
                        **asset_fields,
                        "asset_paths": [image_path],
                        "asset_types": [entity_type],
                        "chunk_id": f"{visual_parent_id}::asset",
                        "chunk_type": "visual_asset_chunk",
                        "parent_id": visual_parent_id,
                    },
                )
            )

        context_parts = [part for part in (candidate.context_before, candidate.context_after) if part]
        if context_parts:
            visual_child_ids.append(f"{visual_parent_id}::context")
            blocks.append(
                ContentBlock(
                    text="\n\n".join(context_parts),
                    type="text",
                    page=candidate.page_no,
                    source_path=str(pdf_path),
                    metadata={
                        "chunk_id": f"{visual_parent_id}::context",
                        "document_type": "pdf",
                        "chunk_type": "visual_context_chunk",
                        "entity_type": "visual_context",
                        "linked_entity_id": candidate.entity_id,
                        "linked_entity_type": entity_type,
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "document_title": document.metadata.get("document_title", pdf_path.stem),
                        "chapter_number": candidate.chapter_number,
                        "chapter_title": candidate.chapter_title,
                        "section_title": candidate.section_title,
                        "subsection_title": candidate.subsection_title,
                        "page_no": candidate.page_no,
                        "page_start": candidate.page_no,
                        "page_end": candidate.page_no,
                        "parent_id": visual_parent_id,
                        "preserve_child_text": True,
                        "contains_figure": False,
                        "contains_chart": False,
                        "contains_diagram": False,
                        "contains_image": False,
                        "contains_map": False,
                        "contains_table": False,
                    },
                )
            )

        for block in blocks[-len(visual_child_ids):]:
            if str(block.metadata.get("parent_id") or "") == visual_parent_id:
                block.metadata["child_ids"] = list(visual_child_ids)

    for image in images:
        resolved_path = str(Path(image.image_path).resolve())
        if resolved_path in used_assets:
            continue
        candidate_entity_id = str(image.metadata.get("entity_id") or normalize_visual_entity_id(image.type, str(image.page or image.element_id)))
        entity_type = _visual_kind_from_entity(image.type, str(image.metadata.get("source_label") or ""))
        if (image.page, entity_type) in covered_page_types:
            continue
        visual_parent_id = (
            f"pdf::{file_hash}::visual::{candidate_entity_id}"
            f"::page::{image.page or 0}"
            f"::{_stable_suffix(resolved_path, image.metadata.get('source_label'))}"
        )
        description = descriptions_by_path.get(str(image.image_path))
        description_text = description.description if description is not None else ""
        asset_fields = _visual_path_fields(entity_type, resolved_path)
        chapter_context = _page_chapter_context(document, image.page)
        validation = validate_asset_path(resolved_path, _asset_validation_type(entity_type, resolved_path))
        blocks.append(
            ContentBlock(
                text=str(image.metadata.get("source_label") or candidate_entity_id),
                type="text",
                page=image.page,
                source_path=str(pdf_path),
                metadata={
                    "chunk_id": f"{visual_parent_id}::caption",
                    "document_type": "pdf",
                    "chunk_type": "visual_caption_chunk",
                    "entity_type": entity_type,
                    "entity_id": candidate_entity_id,
                    "entity_ids": [candidate_entity_id],
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "document_title": document.metadata.get("document_title", pdf_path.stem),
                    "page_no": image.page,
                    "page_start": image.page,
                    "page_end": image.page,
                    **chapter_context,
                    "visual_title": str(image.metadata.get("source_label") or candidate_entity_id),
                    "caption_text": str(image.metadata.get("source_label") or candidate_entity_id),
                    "parent_id": visual_parent_id,
                    "child_ids": [f"{visual_parent_id}::caption", f"{visual_parent_id}::asset"],
                    "linked_entity_id": candidate_entity_id,
                    "linked_entity_type": entity_type,
                    "preserve_child_text": True,
                    "docling_label": str(image.metadata.get("category") or ""),
                    "bbox": image.coordinates,
                    "contains_figure": entity_type in {"figure", "chart", "diagram"},
                    "contains_chart": entity_type == "chart",
                    "contains_diagram": entity_type == "diagram",
                    "contains_image": True,
                    "contains_map": entity_type == "map",
                    "contains_table": False,
                },
            )
        )
        blocks.append(
            ContentBlock(
                text=description_text.strip() or f"Verified visual asset for {candidate_entity_id}.",
                type=entity_type if entity_type in {"chart", "diagram", "figure", "image", "map", "table"} else "image",
                page=image.page,
                source_path=str(pdf_path),
                metadata={
                    "chunk_id": f"{visual_parent_id}::asset",
                    "document_type": "pdf",
                    "chunk_type": "visual_asset_chunk",
                    "entity_type": entity_type,
                    "entity_id": candidate_entity_id,
                    "entity_ids": [candidate_entity_id],
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "document_title": document.metadata.get("document_title", pdf_path.stem),
                    "page_no": image.page,
                    "page_start": image.page,
                    "page_end": image.page,
                    **chapter_context,
                    "visual_title": str(image.metadata.get("source_label") or candidate_entity_id),
                    "caption_text": str(image.metadata.get("source_label") or candidate_entity_id),
                    "parent_id": visual_parent_id,
                    "child_ids": [f"{visual_parent_id}::caption", f"{visual_parent_id}::asset"],
                    "linked_entity_id": candidate_entity_id,
                    "linked_entity_type": entity_type,
                    "preserve_child_text": True,
                    "asset_exists": validation.ok,
                    "asset_validation_status": "allowed" if validation.ok else "blocked",
                    "asset_validation_reason": validation.reason,
                    "docling_label": str(image.metadata.get("category") or ""),
                    "bbox": image.coordinates,
                    "asset_paths": [resolved_path],
                    "asset_types": [entity_type],
                    **asset_fields,
                    "contains_figure": entity_type in {"figure", "chart", "diagram"},
                    "contains_chart": entity_type == "chart",
                    "contains_diagram": entity_type == "diagram",
                    "contains_image": True,
                    "contains_map": entity_type == "map",
                    "contains_table": False,
                },
            )
        )
    return blocks


def build_pdf_document(pdf_path: Path, markdown: str, parser_name: str = "docling") -> EnrichedDocument:
    structure = build_pdf_structure(pdf_path, markdown, parser_name=parser_name)
    blocks = []
    if structure.outline_block is not None:
        blocks.append(structure.outline_block)
    blocks.extend(structure.chapter_blocks)
    blocks.extend(structure.text_blocks)
    return EnrichedDocument(
        source_path=str(pdf_path),
        markdown=markdown.strip() + "\n",
        blocks=blocks,
        metadata={
            **structure.metadata,
            "source_type": "pdf",
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "document_title": structure.document_title,
            "headings": structure.headings,
            "visual_candidates": [asdict(candidate) for candidate in structure.visual_candidates],
        },
    )
