from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ingestion.entity_metadata import enrich_records_with_cross_references
from ingestion.parent_child import attach_parent_context


load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "conversational_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
SCROLL_BATCH_SIZE = int(os.getenv("QDRANT_MIGRATION_SCROLL_BATCH_SIZE", "256"))

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    logger.info("Connecting to Qdrant at %s", QDRANT_URL)
    return QdrantClient(url=QDRANT_URL)


def scroll_all_points(client: QdrantClient) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata") or {})
            text = str(payload.get("text") or payload.get("page_content") or "").strip()
            records.append(
                {
                    "id": point.id,
                    "text": text,
                    "source": str(payload.get("source") or metadata.get("source_file") or "unknown"),
                    "metadata": metadata,
                }
            )
        logger.info("Scanned %s existing Qdrant points", len(records))
        if offset is None:
            return records


def changed_metadata(original: dict[str, Any], updated: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "entity_id",
        "entity_ids",
        "cross_reference",
        "cross_references",
        "contains_table",
        "parent_id",
        "parent_text",
        "chunk_role",
    )
    return {
        key: updated[key]
        for key in keys
        if key in updated and updated.get(key) != original.get(key)
    }


def ensure_payload_indexes(client: QdrantClient) -> None:
    from qdrant_client import models

    for field_name in (
        "metadata.entity_id",
        "metadata.entity_ids",
        "metadata.cross_reference",
        "metadata.cross_references",
        "metadata.parent_id",
    ):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logger.debug("Payload index %s already exists or could not be created: %s", field_name, exc)


def migrate(dry_run: bool = False) -> int:
    client = get_qdrant_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(f"Qdrant collection does not exist: {COLLECTION_NAME}")

        records = scroll_all_points(client)
        enriched_records = attach_parent_context(enrich_records_with_cross_references(records))
        updated_count = 0

        for original, enriched in zip(records, enriched_records):
            metadata_patch = changed_metadata(
                dict(original.get("metadata") or {}),
                dict(enriched.get("metadata") or {}),
            )
            if not metadata_patch:
                continue

            updated_count += 1
            logger.info("Updating point %s with metadata %s", original["id"], metadata_patch)
            if dry_run:
                continue
            client.set_payload(
                collection_name=COLLECTION_NAME,
                points=[original["id"]],
                payload={"metadata": {**dict(original.get("metadata") or {}), **metadata_patch}},
                wait=True,
            )

        if not dry_run:
            ensure_payload_indexes(client)
        logger.info(
            "Cross-reference migration complete. scanned=%s updated=%s dry_run=%s",
            len(records),
            updated_count,
            dry_run,
        )
        print(
            f"Cross-reference migration complete: scanned={len(records)}, "
            f"updated={updated_count}, dry_run={dry_run}"
        )
        return updated_count
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill table/figure cross-reference metadata in Qdrant without modifying vectors."
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned updates without writing payload changes.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
