from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.multimodal_assets import ASSET_FIELDS, enrich_chunk_metadata


@dataclass(slots=True)
class ChunkPayload:
    """Canonical Qdrant payload for multimodal-enriched chunks."""

    text: str
    chunk_id: str
    source_file: str = ""
    source_path: str = ""
    document_type: str = ""
    page: int | None = None
    chapter_number: str = ""
    chapter_title: str = ""
    section_title: str = ""
    subsection_title: str = ""
    h1: str = ""
    h2: str = ""
    h3: str = ""
    chunk_type: str = ""
    entity_type: str = ""
    entity_id: str = ""
    entity_ids: list[str] = field(default_factory=list)
    visual_title: str = ""
    caption_text: str = ""
    source_note: str = ""
    linked_entity_id: str = ""
    linked_entity_type: str = ""
    contains_chart: bool = False
    contains_figure: bool = False
    contains_image: bool = False
    contains_csv: bool = False
    contains_table: bool = False
    contains_diagram: bool = False
    contains_map: bool = False
    contains_csv_semantic_sentence: bool = False
    image_reference: str = ""
    visual_type: str = ""
    language: str = ""
    asset_fields: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "document_type": self.document_type,
            "page": self.page,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "section_title": self.section_title,
            "subsection_title": self.subsection_title,
            "h1": self.h1,
            "h2": self.h2,
            "h3": self.h3,
            "chunk_type": self.chunk_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "entity_ids": self.entity_ids,
            "visual_title": self.visual_title,
            "caption_text": self.caption_text,
            "source_note": self.source_note,
            "linked_entity_id": self.linked_entity_id,
            "linked_entity_type": self.linked_entity_type,
            "contains_chart": self.contains_chart,
            "contains_figure": self.contains_figure,
            "contains_image": self.contains_image,
            "contains_csv": self.contains_csv,
            "contains_table": self.contains_table,
            "contains_diagram": self.contains_diagram,
            "contains_map": self.contains_map,
            "contains_csv_semantic_sentence": self.contains_csv_semantic_sentence,
            "image_reference": self.image_reference,
            "visual_type": self.visual_type,
            "language": self.language,
            **_json_safe(self.asset_fields),
            "metadata": _json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in ("", None, {})}


def normalize_payload(text: str, metadata: dict[str, Any]) -> ChunkPayload:
    """Map ingestion/embedding metadata into a stable Qdrant payload schema."""

    metadata = enrich_chunk_metadata(metadata, text)
    source = str(metadata.get("source") or metadata.get("source_path") or "")
    source_file = str(
        metadata.get("source_file")
        or metadata.get("source_files")
        or (Path(source).name if source else "")
    )
    document_type = str(
        metadata.get("document_type")
        or metadata.get("source_type")
        or Path(source_file).suffix.lstrip(".")
        or "text"
    ).lower()
    page = metadata.get("page_no", metadata.get("page", metadata.get("source_page")))
    try:
        page_value = int(page) if page not in ("", None) else None
    except (TypeError, ValueError):
        page_value = None

    image_reference = str(
        metadata.get("image_reference")
        or metadata.get("image_path")
        or metadata.get("image_local_path")
        or ""
    )

    return ChunkPayload(
        text=text,
        chunk_id=str(metadata.get("chunk_id") or metadata.get("id") or ""),
        source_file=source_file,
        source_path=source,
        document_type=document_type,
        page=page_value,
        chapter_number=str(metadata.get("chapter_number") or ""),
        chapter_title=str(metadata.get("chapter_title") or ""),
        section_title=str(metadata.get("section_title") or metadata.get("section") or metadata.get("h1") or ""),
        subsection_title=str(metadata.get("subsection_title") or metadata.get("h3") or ""),
        h1=str(metadata.get("h1") or ""),
        h2=str(metadata.get("h2") or ""),
        h3=str(metadata.get("h3") or ""),
        chunk_type=str(metadata.get("chunk_type") or ""),
        entity_type=str(metadata.get("entity_type") or ""),
        entity_id=str(metadata.get("entity_id") or ""),
        entity_ids=list(metadata.get("entity_ids") or [] if not isinstance(metadata.get("entity_ids"), str) else [metadata.get("entity_ids")]),
        visual_title=str(metadata.get("visual_title") or ""),
        caption_text=str(metadata.get("caption_text") or ""),
        source_note=str(metadata.get("source_note") or ""),
        linked_entity_id=str(metadata.get("linked_entity_id") or ""),
        linked_entity_type=str(metadata.get("linked_entity_type") or ""),
        contains_chart=bool(metadata.get("contains_chart") or metadata.get("contains_chart_description")),
        contains_figure=bool(metadata.get("contains_figure")),
        contains_image=bool(metadata.get("contains_image") or metadata.get("image_path")),
        contains_csv=bool(metadata.get("contains_csv") or document_type == "csv"),
        contains_table=bool(metadata.get("contains_table")),
        contains_diagram=bool(metadata.get("contains_diagram")),
        contains_map=bool(metadata.get("contains_map")),
        contains_csv_semantic_sentence=bool(
            metadata.get("contains_csv_semantic_sentence") or document_type == "csv"
        ),
        image_reference=image_reference,
        visual_type=str(metadata.get("visual_type") or metadata.get("type") or ""),
        language=str(metadata.get("language") or ""),
        asset_fields={key: metadata.get(key) for key in ASSET_FIELDS if metadata.get(key) not in ("", None, [], {})},
        metadata=metadata,
    )


def qdrant_payload_indexes() -> dict[str, str]:
    """Payload fields to index for common enterprise retrieval filters."""

    return {
        "chunk_id": "keyword",
        "source_file": "keyword",
        "document_type": "keyword",
        "metadata.document_type": "keyword",
        "page": "integer",
        "chapter_number": "keyword",
        "chapter_title": "text",
        "section_title": "text",
        "subsection_title": "text",
        "h1": "text",
        "h2": "text",
        "h3": "text",
        "chunk_type": "keyword",
        "entity_type": "keyword",
        "entity_id": "keyword",
        "entity_ids": "keyword",
        "visual_title": "text",
        "caption_text": "text",
        "linked_entity_id": "keyword",
        "linked_entity_type": "keyword",
        "contains_chart": "bool",
        "contains_figure": "bool",
        "contains_image": "bool",
        "contains_csv": "bool",
        "contains_table": "bool",
        "contains_diagram": "bool",
        "contains_map": "bool",
        "contains_csv_semantic_sentence": "bool",
        "csv_path": "keyword",
        "table_csv_path": "keyword",
        "image_path": "keyword",
        "table_image_path": "keyword",
        "figure_image_path": "keyword",
        "chart_image_path": "keyword",
        "diagram_image_path": "keyword",
        "visual_type": "keyword",
        "language": "keyword",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
