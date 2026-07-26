from __future__ import annotations

import argparse
import logging

from qdrant_client.http import models

from embeddings.embedding_model import EmbeddingModelSettings
from vectordb.create_collection import create_collection, create_payload_indexes
from vectordb.qdrant_client_manager import QdrantSettings, get_qdrant_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the Qdrant collection for BGE-M3 RAG retrieval.")
    parser.add_argument("--collection", default=None, help="Collection name. Defaults to QDRANT_COLLECTION.")
    parser.add_argument("--vector-size", type=int, default=None, help="Embedding dimension. Defaults to BGE-M3 config.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    vector_size = args.vector_size or EmbeddingModelSettings().embedding_dimension
    settings = QdrantSettings()
    collection_name = args.collection or settings.collection_name
    create_collection(
        vector_size=vector_size,
        collection_name=collection_name,
        recreate=args.recreate,
    )

    client = get_qdrant_client(settings)
    logger.info("Initializing payload indexes on collection %s", collection_name)
    for field_name in ("document_type", "metadata.document_type"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload index ready: %s", field_name)
        except Exception as exc:
            logger.debug("Payload index %s skipped or already exists: %s", field_name, exc)
    create_payload_indexes(client=client, collection_name=collection_name)


if __name__ == "__main__":
    main()
