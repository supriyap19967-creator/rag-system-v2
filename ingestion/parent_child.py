from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def _stable_parent_id(source: str, group_key: str) -> str:
    digest = hashlib.sha256(f"{source}|{group_key}".encode("utf-8")).hexdigest()
    return f"parent-{digest[:24]}"


def _parent_group(record: dict[str, Any]) -> tuple[str, str]:
    metadata = dict(record.get("metadata") or {})
    source = str(record.get("source") or metadata.get("source_file") or "unknown")
    document_type = str(metadata.get("document_type") or "text")
    page = metadata.get("page")
    if isinstance(page, int):
        return source, f"page:{page}"
    if document_type == "csv":
        return source, f"csv-row:{metadata.get('row_id', metadata.get('chunk_id', 'unknown'))}"
    # Existing PDF indexes do not always retain page coordinates. Keep the
    # fallback bounded rather than duplicating an entire report per child.
    return source, f"bounded:{metadata.get('chunk_index', metadata.get('chunk_id', 'unknown'))}"


def attach_parent_context(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach page-level parent payloads to existing embedding-ready child records."""

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        key = _parent_group(record)
        text = str(record.get("text") or "").strip()
        if text and text not in grouped[key]:
            grouped[key].append(text)

    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        metadata = dict(item.get("metadata") or {})
        source, group_key = _parent_group(item)
        parent_text = "\n\n".join(grouped[(source, group_key)]).strip()
        metadata["parent_id"] = _stable_parent_id(source, group_key)
        metadata["parent_text"] = parent_text or str(item.get("text") or "").strip()
        metadata["chunk_role"] = "child"
        item["metadata"] = metadata
        enriched.append(item)
    return enriched
