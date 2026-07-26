from __future__ import annotations

import re
from typing import Iterable

from app.multimodal_assets import build_asset_registry, enrich_chunk_metadata
from ingestion.schemas import Chunk, ContentBlock, EnrichedDocument


BOUNDARY_PATTERN = re.compile(r"(?m)^(#{1,6}\s+.+|\[CHART DESCRIPTION\]|\[/CHART DESCRIPTION\])$")
MARKDOWN_HEADING_PATTERN = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
CHAPTER_HEADING_PATTERN = re.compile(
    r"^\s*(?:chapter|ch\.?)\s+(?P<number>[A-Za-z0-9IVXLCDM]+)"
    r"(?:\s*[:.\-–—]\s*|\s+)?(?P<title>.*)$",
    flags=re.IGNORECASE,
)
NUMBERED_CHAPTER_HEADING_PATTERN = re.compile(
    r"^\s*(?P<number>\d+)\s+"
    r"(?P<title>[A-Z][A-Za-z0-9,;:'\"()\-–— ]{3,})$"
)


def _split_markdown_sections(markdown: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    in_chart_block = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == "[CHART DESCRIPTION]":
            if current:
                sections.append("\n".join(current).strip())
                current = []
            in_chart_block = True
        current.append(line)
        if stripped == "[/CHART DESCRIPTION]":
            sections.append("\n".join(current).strip())
            current = []
            in_chart_block = False
            continue
        if not in_chart_block and stripped.startswith("#") and len(current) > 1:
            heading = current.pop()
            sections.append("\n".join(current).strip())
            current = [heading]
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _clean_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().strip("#").strip())


def _chapter_from_heading(title: str) -> tuple[str, str]:
    heading = _clean_heading_text(title)
    match = CHAPTER_HEADING_PATTERN.match(heading)
    if match:
        chapter_number = match.group("number").strip()
        chapter_title = _clean_heading_text(match.group("title") or heading)
        return chapter_number, chapter_title or heading

    match = NUMBERED_CHAPTER_HEADING_PATTERN.match(heading)
    if match:
        return match.group("number").strip(), _clean_heading_text(match.group("title"))

    return "", ""


def _heading_metadata_for_section(section: str, active_headings: dict[int, str]) -> tuple[dict[str, str], dict[int, str]]:
    headings = dict(active_headings)
    for line in section.splitlines():
        match = MARKDOWN_HEADING_PATTERN.match(line.strip())
        if not match:
            continue
        level = len(match.group("level"))
        title = _clean_heading_text(match.group("title"))
        headings = {key: value for key, value in headings.items() if key < level}
        headings[level] = title

    metadata: dict[str, str] = {}
    for level in range(1, 4):
        if headings.get(level):
            metadata[f"h{level}"] = headings[level]

    chapter_number = ""
    chapter_title = ""
    for level in sorted(headings):
        chapter_number, chapter_title = _chapter_from_heading(headings[level])
        if chapter_number:
            break
    if chapter_number:
        metadata["chapter_number"] = chapter_number
        metadata["chapter_title"] = chapter_title

    section_title = headings.get(max(headings)) if headings else ""
    if section_title:
        metadata["section_title"] = section_title
        metadata["section"] = section_title
        metadata["section_header"] = section_title
    return metadata, headings


def _recursive_split(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    separators = ["\n\n", "\n", ". ", " "]
    for separator in separators:
        parts = text.split(separator)
        if len(parts) == 1:
            continue
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current}{separator}{part}"
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(_recursive_split(current.strip(), chunk_size))
                current = part
        if current:
            chunks.extend(_recursive_split(current.strip(), chunk_size))
        return chunks
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _with_overlap(chunks: Iterable[str], overlap: int) -> list[str]:
    result: list[str] = []
    previous = ""
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if previous and overlap > 0:
            prefix = previous[-overlap:].strip()
            if prefix and not chunk.startswith(prefix):
                chunk = f"{prefix}\n\n{chunk}"
        result.append(chunk)
        previous = chunk
    return result


class MarkdownChunker:
    """Markdown-aware recursive chunker that keeps tables and chart blocks intact."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 180) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: EnrichedDocument) -> list[Chunk]:
        if (
            document.metadata.get("source_type") in {"csv", "extracted_table_csv"}
            or any(block.metadata.get("chunk_type") for block in document.blocks)
        ) and document.blocks:
            asset_registry = build_asset_registry()
            return [
                Chunk(
                    text=block.text,
                    metadata=enrich_chunk_metadata(
                        {
                            "source": document.source_path,
                            "source_path": document.source_path,
                            "source_type": document.metadata.get("source_type"),
                            "source_file": document.metadata.get("source_file"),
                            "document_type": document.metadata.get("source_type"),
                            "chunk_index": index,
                            **dict(document.metadata),
                            **dict(block.metadata),
                        },
                        block.text,
                        asset_registry,
                    ),
                )
                for index, block in enumerate(document.blocks, start=1)
                if str(block.text or "").strip()
            ]

        sections = _split_markdown_sections(document.markdown)
        raw_chunks: list[tuple[str, dict[str, str]]] = []
        active_headings: dict[int, str] = {}
        asset_registry = build_asset_registry()
        for section in sections:
            hierarchy_metadata, active_headings = _heading_metadata_for_section(section, active_headings)
            if section.startswith("[CHART DESCRIPTION]") or "|" in section:
                raw_chunks.append((section, hierarchy_metadata))
            else:
                raw_chunks.extend((chunk, hierarchy_metadata) for chunk in _recursive_split(section, self.chunk_size))

        chunk_texts = _with_overlap((text for text, _metadata in raw_chunks), self.chunk_overlap)
        chunk_metadata = [metadata for _text, metadata in raw_chunks]
        return [
            Chunk(
                text=text,
                metadata=enrich_chunk_metadata(
                    {
                        "source": document.source_path,
                        "source_path": document.source_path,
                        "source_type": document.metadata.get("source_type"),
                        "source_file": document.metadata.get("source_file"),
                        "document_type": document.metadata.get("source_type"),
                        "chunk_index": index,
                        "contains_chart_description": "[CHART DESCRIPTION]" in text,
                        **(chunk_metadata[index] if index < len(chunk_metadata) else {}),
                    },
                    text,
                    asset_registry,
                ),
            )
            for index, text in enumerate(chunk_texts)
        ]
