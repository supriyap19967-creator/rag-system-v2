from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.multimodal_assets import normalize_entity_id


def safe_token(value: str, fallback: str = "visual") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_") or fallback


def entity_token_from_label(label: str, fallback: str = "visual") -> str:
    normalized = normalize_entity_id(label)
    if normalized:
        return safe_token(normalized.replace(" ", "_"))
    match = re.search(r"[\d]+(?:\.[\d]+)*[A-Za-z]?", str(label or ""))
    if match:
        return safe_token(match.group(0))
    return safe_token(label, fallback=fallback)


def canonical_flat_image_path(
    root_dir: Path,
    source_path: Path,
    *,
    page_number: int | None,
    visual_type: str,
    entity_label: str,
) -> Path:
    """Copy or move an extracted crop into the canonical flat asset directory."""

    root_dir.mkdir(parents=True, exist_ok=True)
    page_token = page_number if page_number is not None else "unknown"
    type_token = safe_token(visual_type.title() if visual_type else "Figure", fallback="Figure")
    entity_token = entity_token_from_label(entity_label, fallback=f"idx_{source_path.stem}")
    suffix = source_path.suffix.lower() if source_path.suffix else ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    destination = root_dir / f"page_{page_token}_{type_token}_{entity_token}{suffix}"
    source_resolved = source_path.expanduser().resolve()
    destination_resolved = destination.resolve()
    if source_resolved != destination_resolved:
        if destination_resolved.exists():
            destination_resolved.unlink()
        shutil.copy2(source_resolved, destination_resolved)
    return destination_resolved


def absolute_asset_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())
