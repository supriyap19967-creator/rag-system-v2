from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


ENTITY_PATTERN = re.compile(r"\b(?P<kind>Table|Figure)\s+(?P<identifier>[Oo0]?\s*\.?\s*\d+(?:\s*\.\s*\d+)*)", re.IGNORECASE)


def normalize_entity_id(kind: str, identifier: str) -> str:
    normalized = re.sub(r"\s+", "", identifier or "").upper().replace("0.", "O.")
    if re.fullmatch(r"[O0]\d+", normalized):
        normalized = f"O.{normalized[1:]}"
    prefix = "Table" if kind.lower() == "table" else "Figure"
    return f"{prefix} {normalized}"


def extract_entity_ids(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in ENTITY_PATTERN.finditer(text or ""):
        entity_id = normalize_entity_id(match.group("kind"), match.group("identifier"))
        if entity_id.lower() not in seen:
            seen.add(entity_id.lower())
            entities.append(entity_id)
    return entities


def _metadata_entity_ids(record: dict[str, Any]) -> list[str]:
    metadata = dict(record.get("metadata") or {})
    values = [
        metadata.get("entity_id"),
        metadata.get("figure_id"),
        metadata.get("source_label"),
    ]
    entities: list[str] = []
    for value in values:
        entities.extend(extract_entity_ids(str(value or "")))
    entities.extend(extract_entity_ids(str(record.get("text") or "")))
    return list(dict.fromkeys(entities))


def enrich_records_with_cross_references(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach normalized entity IDs and table/figure companion links before Qdrant upsert."""

    entities_by_suffix: dict[str, set[str]] = defaultdict(set)
    entities_by_page: dict[tuple[str, int], set[str]] = defaultdict(set)
    record_entities: list[list[str]] = []
    last_table_by_source: dict[str, str] = {}
    for record in records:
        entities = _metadata_entity_ids(record)
        source = str(record.get("source") or (record.get("metadata") or {}).get("source_file") or "")
        text = str(record.get("text") or "")
        explicit_tables = [entity for entity in entities if entity.startswith("Table ")]
        if explicit_tables:
            last_table_by_source[source] = explicit_tables[-1]
        elif "|" in text and source in last_table_by_source:
            entities.append(last_table_by_source[source])
            entities = list(dict.fromkeys(entities))
        record_entities.append(entities)
        for entity_id in entities:
            _, identifier = entity_id.split(" ", 1)
            entities_by_suffix[identifier].add(entity_id)
            page = (record.get("metadata") or {}).get("page")
            if isinstance(page, int):
                entities_by_page[(source, page)].add(entity_id)

    enriched: list[dict[str, Any]] = []
    for record, entities in zip(records, record_entities):
        item = dict(record)
        metadata = dict(item.get("metadata") or {})
        text = str(item.get("text") or "")
        if entities:
            metadata["entity_id"] = entities[0]
            metadata["entity_ids"] = entities

        references: list[str] = []
        for entity_id in entities:
            kind, identifier = entity_id.split(" ", 1)
            companion_kind = "Figure" if kind == "Table" else "Table"
            companion = f"{companion_kind} {identifier}"
            if companion in entities_by_suffix.get(identifier, set()):
                references.append(companion)
            page = metadata.get("page")
            if isinstance(page, int):
                references.extend(
                    related
                    for related in entities_by_page.get((str(item.get("source") or metadata.get("source_file") or ""), page), set())
                    if related != entity_id and related.startswith(f"{companion_kind} ")
                )

        if references:
            references = list(dict.fromkeys(references))
            metadata["cross_reference"] = references[0]
            metadata["cross_references"] = references

        if metadata.get("document_type") == "pdf" and "|" in text:
            metadata["contains_table"] = True

        item["metadata"] = metadata
        enriched.append(item)
    return enriched
