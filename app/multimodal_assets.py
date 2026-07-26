from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_ASSET_DIRS = (
    PROJECT_ROOT / "assets" / "extracted_tables",
    PROJECT_ROOT / "assets" / "extracted_images",
    PROJECT_ROOT / "Data" / "Pdf",
    PROJECT_ROOT / "Data" / "csv",
    PROJECT_ROOT / "extracted_charts",
    PROJECT_ROOT / "extracted_images",
)
TABLE_EXTENSIONS = {".csv", ".xlsx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSIONS = {".pdf"}
CSV_EXTENSIONS = {".csv"}
ENTITY_PATTERN = re.compile(
    r"\b(?P<kind>table|figure|fig\.?|chart|image|diagram|map|box|spotlight)\s*[_\-\s]?(?P<identifier>[A-Za-z]?\d+(?:\.\d+)*)\b",
    flags=re.IGNORECASE,
)
PAGE_PATTERN = re.compile(r"\bpage[_\-\s]?(?P<page>\d+)\b", flags=re.IGNORECASE)
MARKDOWN_TABLE_PATTERN = re.compile(r"^\s*\|.+\|\s*$", flags=re.MULTILINE)


ASSET_FIELDS = (
    "image_path",
    "image_paths",
    "figure_image_path",
    "figure_image_paths",
    "chart_image_path",
    "chart_image_paths",
    "diagram_image_path",
    "diagram_image_paths",
    "table_csv_path",
    "table_csv_paths",
    "csv_path",
    "csv_paths",
    "table_image_path",
    "table_image_paths",
    "asset_paths",
    "asset_types",
    "entity_id",
    "entity_ids",
    "entity_type",
    "contains_table",
    "contains_figure",
    "contains_chart",
    "contains_image",
    "contains_csv",
    "contains_diagram",
    "contains_map",
)


@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    asset_type: str
    source_file: str
    page_no: int | None
    entity_id: str
    absolute_path: str
    relative_path: str
    normalized_path: str
    exists_on_disk: bool
    file_extension: str


@dataclass(frozen=True, slots=True)
class PathValidationResult:
    ok: bool
    path: str = ""
    layer: str = "Layer 4 Asset Path Validation"
    reason: str = ""
    action: str = "blocked"
    asset_type: str = ""


@dataclass(frozen=True, slots=True)
class AssetResolution:
    ok: bool
    asset_type: str = ""
    path: str = ""
    renderer: str = ""
    reason: str = ""
    validation: PathValidationResult | None = None
    candidates: list[str] = field(default_factory=list)


def normalize_entity_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(\d+)[\-_](?=\d)", r"\1.", text)
    match = ENTITY_PATTERN.search(text.replace("_", " "))
    if not match:
        return re.sub(r"[^A-Za-z0-9.]+", "_", text).strip("_")
    kind = match.group("kind").lower()
    if kind.startswith("fig"):
        kind = "Figure"
    elif kind == "chart":
        kind = "Chart"
    elif kind == "diagram":
        kind = "Diagram"
    elif kind == "map":
        kind = "Map"
    elif kind == "image":
        kind = "Image"
    elif kind == "box":
        kind = "Box"
    elif kind == "spotlight":
        kind = "Spotlight"
    else:
        kind = "Table"
    return f"{kind}_{match.group('identifier').upper()}"


def entity_ids_from_text(text: object) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for match in ENTITY_PATTERN.finditer(str(text or "").replace("_", " ")):
        entity_id = normalize_entity_id(f"{match.group('kind')} {match.group('identifier')}")
        if entity_id and entity_id.lower() not in seen:
            seen.add(entity_id.lower())
            output.append(entity_id)
    return output


def page_no_from_text(text: object) -> int | None:
    match = PAGE_PATTERN.search(str(text or ""))
    if not match:
        return None
    try:
        return int(match.group("page"))
    except ValueError:
        return None


def requested_asset_type(query: str) -> str:
    lowered = str(query or "").lower()
    if re.search(r"\b(table|tabular|csv|spreadsheet|rows?|columns?)\b", lowered):
        return "table"
    if re.search(r"\b(figure|fig\.?|chart|graph|image|visual|diagram|map)\b", lowered):
        return "image"
    if lowered.endswith(".csv") or "csv" in lowered:
        return "csv"
    return ""


def _asset_type_for_path(path: Path) -> str:
    name = path.stem.lower()
    suffix = path.suffix.lower()
    if suffix in CSV_EXTENSIONS:
        return "source_csv" if "Data\\csv" in str(path) or "Data/csv" in str(path) else "table_csv"
    if suffix in PDF_EXTENSIONS:
        return "source_pdf"
    if suffix in IMAGE_EXTENSIONS:
        if "table" in name:
            return "table_image"
        if "chart" in name:
            return "chart_image"
        if "diagram" in name:
            return "diagram_image"
        return "figure_image"
    return "unknown"


def build_asset_registry(root: Path | None = None) -> list[AssetRecord]:
    root = (root or PROJECT_ROOT).resolve()
    records: list[AssetRecord] = []
    for asset_dir in APPROVED_ASSET_DIRS:
        directory = asset_dir if asset_dir.is_absolute() else root / asset_dir
        if not directory.exists() or not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in TABLE_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS | CSV_EXTENSIONS:
                continue
            resolved = path.resolve()
            entity_id = normalize_entity_id(path.stem)
            page_no = page_no_from_text(path.stem)
            relative = str(resolved.relative_to(root)) if _is_relative_to(resolved, root) else str(resolved)
            asset_id = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()
            records.append(
                AssetRecord(
                    asset_id=asset_id,
                    asset_type=_asset_type_for_path(resolved),
                    source_file=path.name,
                    page_no=page_no,
                    entity_id=entity_id,
                    absolute_path=str(resolved),
                    relative_path=relative,
                    normalized_path=str(resolved),
                    exists_on_disk=resolved.is_file(),
                    file_extension=suffix,
                )
            )
    return records


@lru_cache(maxsize=4)
def _cached_asset_registry(root: str) -> tuple[AssetRecord, ...]:
    return tuple(build_asset_registry(Path(root)))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _as_list(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        return [str(item) for item in value.values() if str(item or "").strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _append_unique(metadata: dict[str, Any], key: str, value: str) -> None:
    if not value:
        return
    values = _as_list(metadata.get(key))
    if value not in values:
        values.append(value)
    metadata[key] = values


def _set_default_path(metadata: dict[str, Any], key: str, value: str) -> None:
    if value and not metadata.get(key):
        metadata[key] = value


def _entity_kind(entity_id: object) -> str:
    normalized = normalize_entity_id(entity_id)
    if normalized.startswith("Table_"):
        return "table"
    if normalized.startswith("Chart_"):
        return "chart"
    if normalized.startswith("Diagram_"):
        return "diagram"
    if normalized.startswith("Map_"):
        return "map"
    if normalized.startswith("Figure_"):
        return "figure"
    return ""


def _entity_matches_record(entity_id: object, record: AssetRecord) -> bool:
    normalized_entity = normalize_entity_id(entity_id)
    if not normalized_entity:
        return False
    normalized_record = normalize_entity_id(record.entity_id)
    if normalized_entity != normalized_record:
        return False
    entity_kind = _entity_kind(normalized_entity)
    record_kind = "table" if record.asset_type.startswith("table") else "chart" if record.asset_type == "chart_image" else "diagram" if record.asset_type == "diagram_image" else "figure" if record.asset_type == "figure_image" else "csv" if record.asset_type == "source_csv" else ""
    if entity_kind == "table":
        return record_kind in {"table", "csv"}
    if entity_kind == "chart":
        return record_kind == "chart"
    if entity_kind == "diagram":
        return record_kind == "diagram"
    if entity_kind == "map":
        return record_kind == "figure"
    if entity_kind == "figure":
        return record_kind == "figure"
    return False


def _set_text_partitions(enriched: dict[str, Any], text: str, document_type: str) -> None:
    clean_text = str(text or "").strip()
    if not clean_text:
        return
    if document_type == "csv":
        enriched["strict_extracted_text"] = clean_text
        enriched.setdefault("description", clean_text)
        return
    if enriched.get("contains_table") and not enriched.get("strict_extracted_text"):
        enriched.setdefault("fallback_nearby_text", clean_text)
        enriched.setdefault("description", clean_text)
        return
    enriched.setdefault("description", clean_text)


def _clear_exclusive_asset_fields(enriched: dict[str, Any], entity_kind: str) -> None:
    exclusive_fields = {
        "table": (
            "image_path",
            "image_paths",
            "figure_image_path",
            "figure_image_paths",
            "chart_image_path",
            "chart_image_paths",
            "diagram_image_path",
            "diagram_image_paths",
        ),
        "figure": (
            "table_csv_path",
            "table_csv_paths",
            "csv_path",
            "csv_paths",
            "table_image_path",
            "table_image_paths",
        ),
        "chart": (
            "table_csv_path",
            "table_csv_paths",
            "csv_path",
            "csv_paths",
            "table_image_path",
            "table_image_paths",
        ),
        "diagram": (
            "table_csv_path",
            "table_csv_paths",
            "csv_path",
            "csv_paths",
            "table_image_path",
            "table_image_paths",
        ),
        "map": (
            "table_csv_path",
            "table_csv_paths",
            "csv_path",
            "csv_paths",
            "table_image_path",
            "table_image_paths",
        ),
    }
    for key in exclusive_fields.get(entity_kind, ()):
        enriched.pop(key, None)

    if entity_kind == "table":
        enriched["contains_table"] = True
        enriched["contains_figure"] = False
        enriched["contains_chart"] = False
    elif entity_kind == "figure":
        enriched["contains_figure"] = True
        enriched["contains_chart"] = False
        enriched["contains_table"] = bool(
            enriched.get("table_csv_path")
            or enriched.get("table_csv_paths")
            or enriched.get("table_image_path")
            or enriched.get("table_image_paths")
        )
    elif entity_kind == "chart":
        enriched["contains_chart"] = True
        enriched["contains_figure"] = False
        enriched["contains_table"] = bool(
            enriched.get("table_csv_path")
            or enriched.get("table_csv_paths")
            or enriched.get("table_image_path")
            or enriched.get("table_image_paths")
        )


def enrich_chunk_metadata(metadata: dict[str, Any], text: str = "", registry: list[AssetRecord] | None = None) -> dict[str, Any]:
    """Attach normalized multimodal metadata without dropping existing values."""

    enriched = dict(metadata or {})
    source = str(enriched.get("source_path") or enriched.get("source") or "")
    source_file = str(enriched.get("source_file") or (Path(source).name if source else ""))
    document_type = str(enriched.get("document_type") or enriched.get("source_type") or Path(source_file).suffix.lstrip(".") or "").lower()
    page_no = enriched.get("page_no", enriched.get("docling_page_no", enriched.get("page", enriched.get("source_page"))))
    try:
        page_int = int(page_no) if page_no not in ("", None) else None
    except (TypeError, ValueError):
        page_int = None

    body = f"{text}\n{source_file}"
    entity_ids = list(dict.fromkeys([*_as_list(enriched.get("entity_ids")), *entity_ids_from_text(body)]))
    if enriched.get("entity_id"):
        entity_ids.insert(0, str(enriched["entity_id"]))
        entity_ids = list(dict.fromkeys(entity_ids))
    if entity_ids:
        enriched["entity_ids"] = entity_ids
        enriched.setdefault("entity_id", entity_ids[0])
        primary_kind = _entity_kind(enriched.get("entity_id"))
        if primary_kind:
            enriched["entity_type"] = primary_kind
            _clear_exclusive_asset_fields(enriched, primary_kind)

    has_markdown_table = bool(MARKDOWN_TABLE_PATTERN.search(text or ""))
    contains_table = bool(
        enriched.get("contains_table")
        or has_markdown_table
        or any(e.lower().startswith("table_") for e in entity_ids)
    )
    contains_figure = bool(
        enriched.get("contains_figure")
        or any(e.lower().startswith("figure_") for e in entity_ids)
        or bool(re.search(r"\b(?:figure|fig\.?)\s+[A-Za-z]?\d+(?:\.\d+)*\b", text or "", flags=re.IGNORECASE))
    )
    contains_chart = bool(
        enriched.get("contains_chart")
        or any(e.lower().startswith("chart_") for e in entity_ids)
        or bool(re.search(r"\bchart\s+[A-Za-z]?\d+(?:\.\d+)*\b", text or "", flags=re.IGNORECASE))
    )
    contains_diagram = bool(
        enriched.get("contains_diagram")
        or any(e.lower().startswith("diagram_") for e in entity_ids)
        or bool(re.search(r"\bdiagram\s+[A-Za-z]?\d+(?:\.\d+)*\b", text or "", flags=re.IGNORECASE))
    )
    contains_map = bool(
        enriched.get("contains_map")
        or any(e.lower().startswith("map_") for e in entity_ids)
        or bool(re.search(r"\bmap\s+[A-Za-z]?\d+(?:\.\d+)*\b", text or "", flags=re.IGNORECASE))
    )
    contains_csv = bool(enriched.get("contains_csv") or document_type == "csv" or str(source_file).lower().endswith(".csv"))
    contains_image = bool(
        enriched.get("contains_image")
        or contains_figure
        or contains_chart
        or contains_diagram
        or contains_map
        or enriched.get("image_path")
        or enriched.get("figure_image_path")
        or enriched.get("chart_image_path")
        or enriched.get("diagram_image_path")
        or enriched.get("table_image_path")
    )
    if contains_table:
        enriched["contains_table"] = True
    if contains_figure:
        enriched["contains_figure"] = True
    if contains_chart:
        enriched["contains_chart"] = True
    if contains_diagram:
        enriched["contains_diagram"] = True
    if contains_map:
        enriched["contains_map"] = True
    if contains_csv:
        enriched["contains_csv"] = True
    if contains_image:
        enriched["contains_image"] = True
    if contains_table and not enriched.get("entity_type"):
        enriched["entity_type"] = "table"
    elif (contains_figure or contains_chart or contains_diagram or contains_map or contains_image) and not enriched.get("entity_type"):
        enriched["entity_type"] = (
            "figure" if contains_figure else
            "chart" if contains_chart else
            "diagram" if contains_diagram else
            "map" if contains_map else
            "image"
        )

    if source_file:
        enriched.setdefault("source_file", source_file)
    if source:
        enriched.setdefault("source_path", source)
    if document_type:
        enriched.setdefault("document_type", document_type)
    if page_int is not None:
        enriched["page_no"] = page_int

    _set_text_partitions(enriched, text, document_type)

    active_registry = registry if registry is not None else list(_cached_asset_registry(str(PROJECT_ROOT)))
    for record in active_registry:
        entity_match = any(_entity_matches_record(entity_id, record) for entity_id in entity_ids)
        if not entity_match:
            continue
        path = record.absolute_path
        _append_unique(enriched, "asset_paths", path)
        _append_unique(enriched, "asset_types", record.asset_type)
        if record.asset_type == "table_csv":
            _set_default_path(enriched, "csv_path", path)
            _set_default_path(enriched, "table_csv_path", path)
            _append_unique(enriched, "csv_paths", path)
            _append_unique(enriched, "table_csv_paths", path)
            enriched["contains_table"] = True
            enriched["contains_csv"] = True
            if document_type == "csv":
                enriched["strict_extracted_text"] = str(text or "").strip()
                enriched["description"] = enriched["strict_extracted_text"]
        elif record.asset_type == "table_image":
            _set_default_path(enriched, "table_image_path", path)
            _append_unique(enriched, "table_image_paths", path)
            enriched["contains_table"] = True
            enriched["contains_image"] = True
        elif record.asset_type in {"figure_image", "chart_image", "diagram_image"}:
            _set_default_path(enriched, "image_path", path)
            _append_unique(enriched, "image_paths", path)
            if record.asset_type == "chart_image":
                target_key = "chart_image_path"
                target_list = "chart_image_paths"
            elif record.asset_type == "diagram_image":
                target_key = "diagram_image_path"
                target_list = "diagram_image_paths"
            else:
                target_key = "figure_image_path"
                target_list = "figure_image_paths"
            _set_default_path(enriched, target_key, path)
            _append_unique(enriched, target_list, path)
            enriched["contains_image"] = True
            enriched["contains_chart"] = bool(enriched.get("contains_chart") or record.asset_type == "chart_image")
            enriched["contains_diagram"] = bool(enriched.get("contains_diagram") or record.asset_type == "diagram_image")
            enriched["contains_figure"] = bool(enriched.get("contains_figure") or record.asset_type == "figure_image")
        elif record.asset_type == "source_csv":
            _set_default_path(enriched, "csv_path", path)
            _append_unique(enriched, "csv_paths", path)
            enriched["contains_csv"] = True

    primary_kind = _entity_kind(enriched.get("entity_id"))
    if primary_kind:
        enriched["entity_type"] = primary_kind
        _clear_exclusive_asset_fields(enriched, primary_kind)
    if enriched.get("strict_extracted_text") and enriched.get("fallback_nearby_text"):
        enriched["description"] = (
            f"[STRICT_EXTRACTED]\n{enriched['strict_extracted_text']}\n\n"
            f"[FALLBACK_NEARBY]\n{enriched['fallback_nearby_text']}"
        )
    return enriched


def validate_asset_path(path_value: object, asset_type: str = "") -> PathValidationResult:
    raw = str(path_value or "").strip()
    if not raw:
        return PathValidationResult(False, reason="asset path was empty or null", asset_type=asset_type)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    approved_dirs = [directory.resolve() for directory in APPROVED_ASSET_DIRS if directory.exists()]
    if not any(_is_relative_to(path, directory) for directory in approved_dirs):
        return PathValidationResult(False, str(path), reason="path is outside the approved asset directories", asset_type=asset_type)
    allowed = IMAGE_EXTENSIONS
    if asset_type in {"table", "table_csv", "csv", "source_csv"}:
        allowed = TABLE_EXTENSIONS | CSV_EXTENSIONS
    elif asset_type in {"pdf", "source_pdf"}:
        allowed = PDF_EXTENSIONS
    elif asset_type in {"table_image", "figure_image", "chart_image", "diagram_image", "image", "figure", "chart", "diagram", "map"}:
        allowed = IMAGE_EXTENSIONS
    if path.suffix.lower() not in allowed:
        return PathValidationResult(False, str(path), reason=f"extension {path.suffix} is not allowed for {asset_type or 'asset'}", asset_type=asset_type)
    if not path.is_file():
        fallback_path = (PROJECT_ROOT / "extracted_images" / path.name).resolve()
        if fallback_path.is_file():
            path = fallback_path
        else:
            return PathValidationResult(False, str(path), reason="asset path does not exist on disk", asset_type=asset_type)
    return PathValidationResult(True, str(path), reason="verified", action="allowed", asset_type=asset_type)


def candidate_asset_paths(chunk: dict[str, Any], requested_type: str = "") -> list[tuple[str, str]]:
    metadata = dict(chunk.get("metadata") or {})
    keys_by_type = {
        "table": ("table_csv_path", "table_csv_paths", "csv_path", "csv_paths", "table_image_path", "table_image_paths", "asset_paths"),
        "csv": ("csv_path", "csv_paths", "table_csv_path", "table_csv_paths", "asset_paths"),
        "image": ("image_path", "image_paths", "figure_image_path", "figure_image_paths", "chart_image_path", "chart_image_paths", "diagram_image_path", "diagram_image_paths", "table_image_path", "table_image_paths", "asset_paths"),
    }
    keys = keys_by_type.get(requested_type) or tuple(ASSET_FIELDS)
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in keys:
        for value in _as_list(metadata.get(key) or chunk.get(key)):
            asset_type = "table" if "table" in key or key.startswith("csv") else "image" if "image" in key or "figure" in key or "chart" in key else requested_type
            if value and value not in seen:
                seen.add(value)
                output.append((value, asset_type))
    return output


def resolve_best_asset(
    user_query: str,
    retrieved_chunks: list[dict[str, Any]],
    registry: list[AssetRecord] | None = None,
) -> AssetResolution:
    requested = requested_asset_type(user_query)
    entity_ids = entity_ids_from_text(user_query)
    candidates: list[tuple[int, str, str]] = []
    for chunk in retrieved_chunks:
        metadata = dict(chunk.get("metadata") or {})
        chunk_entities = [str(item) for item in _as_list(metadata.get("entity_ids"))]
        if metadata.get("entity_id"):
            chunk_entities.append(str(metadata["entity_id"]))
        exact = any(e.lower() == ce.lower() for e in entity_ids for ce in chunk_entities)
        for path, asset_type in candidate_asset_paths(chunk, requested):
            score = 0
            if exact:
                score += 100
            if requested == "table" and asset_type == "table":
                score += 20
            if requested in {"image", ""} and asset_type == "image":
                score += 20
            candidates.append((score, path, asset_type))
    for record in registry or []:
        if requested == "table" and record.asset_type not in {"table_csv", "table_image"}:
            continue
        if requested == "image" and record.asset_type not in {"figure_image", "chart_image"}:
            continue
        exact = any(record.entity_id.lower() == entity.lower() for entity in entity_ids)
        if exact:
            candidates.append((90, record.absolute_path, "table" if record.asset_type.startswith("table") else "image"))
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    checked: list[str] = []
    for _score, path, asset_type in candidates:
        checked.append(path)
        validation = validate_asset_path(path, asset_type)
        if validation.ok:
            renderer = "table_csv" if asset_type == "table" and Path(validation.path).suffix.lower() in CSV_EXTENSIONS else "image"
            return AssetResolution(True, asset_type, validation.path, renderer, "verified asset selected", validation, checked)
    reason = "metadata was missing an asset path" if not candidates else "all candidate asset paths failed Layer 4 validation"
    return AssetResolution(False, requested, reason=reason, candidates=checked)


def preview_csv(path: str, max_rows: int = 20) -> list[list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[:max_rows]
