from __future__ import annotations

import glob
import json
import logging
import os
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync_transcriptions")

def is_valid_transcription(text: str) -> bool:
    """Filter out Vision API error/refusal strings and empty placeholders."""
    if not text or len(text.strip()) < 15:
        return False
    refusal_phrases = [
        "i'm sorry", "cannot extract", "no data", "too blurry",
        "as an ai", "unable to read", "no readable text", "dummy placeholder"
    ]
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in refusal_phrases):
        return False
    return ("|" in text or "SECTION" in text or "SUMMARY" in text or len(text.strip()) > 30)

def generate_deterministic_uuid(name: str) -> str:
    """Generate a stable namespace UUID for Qdrant point keys."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rag_transcription_{name}"))

def main():
    try:
        from vectordb.qdrant_client_manager import get_qdrant_client
        from embeddings.embedding_model import get_embedding_model
        from qdrant_client import models
    except ImportError as e:
        logger.error("Failed to import Qdrant dependencies: %s", e)
        return

    client = get_qdrant_client()
    dense_model = get_embedding_model()
    COLLECTION_NAME = "conversational_rag"

    # Base Supabase storage URL for images
    SUPABASE_BASE_URL = os.getenv("SUPABASE_STORAGE_URL", "https://furnlyvcinfdoctxtkiu.supabase.co/storage/v1/object/public/rag")

    transcriptions_dir = os.path.join(os.getcwd(), "data_cache", "transcriptions")
    if not os.path.exists(transcriptions_dir):
        logger.error("Directory not found: %s", transcriptions_dir)
        return

    json_files = glob.glob(os.path.join(transcriptions_dir, "*.json"))
    logger.info("Found %d transcription JSON files in %s", len(json_files), transcriptions_dir)

    points_to_upsert = []
    valid_count = 0
    skipped_count = 0

    for fpath in json_files:
        stem = Path(fpath).stem.lower()  # e.g., 'figure_4_2' or 'table_2_1'
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            logger.warning("Failed to parse JSON %s: %s", fpath, err)
            skipped_count += 1
            continue

        markdown_text = data.get("markdown_table") or data.get("content") or data.get("text") or ""
        if not is_valid_transcription(markdown_text):
            skipped_count += 1
            continue

        asset_kind = data.get("asset_type") or ("figure" if "fig" in stem or "chart" in stem else "table")
        asset_id = str(data.get("asset_id") or stem.replace("figure_", "").replace("table_", "").replace("chart_", ""))
        id_clean = asset_id.replace(".", "_")

        # Resolve Supabase image URL or local path
        local_img_path = data.get("image_path") or ""
        img_name = os.path.basename(local_img_path) if local_img_path else f"{asset_kind}_{id_clean}.png"
        supabase_img_url = f"{SUPABASE_BASE_URL}/extracted_images/{img_name}"

        # Dual payload key structure (Root keys + Nested metadata keys)
        payload_dict = {
            "chunk_id": f"{asset_kind}_{id_clean}".lower(),
            "entity_id": f"{asset_kind}_{id_clean}".lower(),
            "asset_id": asset_id,
            "asset_type": asset_kind,
            "markdown_table": markdown_text,
            "content": markdown_text,
            "text": markdown_text,
            "page_content": markdown_text,
            "image_path": supabase_img_url,
            "document_type": "pdf_visual" if asset_kind == "figure" else "pdf_table",
            "metadata": {
                "asset_id": asset_id,
                "asset_type": asset_kind,
                "document_type": "pdf_visual" if asset_kind == "figure" else "pdf_table",
                "image_path": supabase_img_url,
                "figure_image_path": supabase_img_url if asset_kind == "figure" else "",
                "table_image_path": supabase_img_url if asset_kind == "table" else "",
                "source_file": "World Development Report 2025.pdf",
            }
        }

        # Generate 384-dim dense embedding
        dense_vector = dense_model.embed_query(markdown_text)

        point_id = generate_deterministic_uuid(stem)
        points_to_upsert.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector
                },
                payload=payload_dict
            )
        )
        valid_count += 1

    logger.info("Validated %d assets for Qdrant Cloud upsert (Skipped %d invalid/refusal assets).", valid_count, skipped_count)

    # Batch upsert points in chunks of 50
    batch_size = 50
    for i in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[i : i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        logger.info("Upserted batch %d/%d (%d points) to Qdrant Cloud.", (i // batch_size) + 1, (len(points_to_upsert) + batch_size - 1) // batch_size, len(batch))

    # Ensure all payload indexes exist on Qdrant Cloud
    for field, schema_type in [
        ("metadata.asset_id", models.PayloadSchemaType.KEYWORD),
        ("metadata.asset_type", models.PayloadSchemaType.KEYWORD),
        ("asset_id", models.PayloadSchemaType.KEYWORD),
        ("asset_type", models.PayloadSchemaType.KEYWORD),
        ("chunk_id", models.PayloadSchemaType.KEYWORD),
        ("entity_id", models.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema_type
            )
            logger.info("Payload index ready: %s", field)
        except Exception as idx_err:
            logger.debug("Index %s status note: %s", field, idx_err)

    logger.info("🎉 SUCCESS: All %d visual asset transcriptions synchronized into Qdrant Cloud payloads!", valid_count)

if __name__ == "__main__":
    main()
