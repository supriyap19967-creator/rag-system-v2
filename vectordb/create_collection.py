from __future__ import annotations

import logging

from vectordb.metadata_schema import qdrant_payload_indexes
from vectordb.qdrant_client_manager import QdrantSettings, get_qdrant_client


logger = logging.getLogger(__name__)


def create_collection(
    vector_size: int,
    collection_name: str | None = None,
    recreate: bool = False,
    client: object | None = None,
    settings: QdrantSettings | None = None,
) -> None:
    """Create a cosine-similarity Qdrant collection for BGE-M3 vectors."""

    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError("Install qdrant-client to create Qdrant collections.") from exc

    settings = settings or QdrantSettings()
    collection_name = collection_name or settings.collection_name
    client = client or get_qdrant_client(settings)

    exists = client.collection_exists(collection_name=collection_name)
    if exists and recreate:
        logger.warning("Recreating Qdrant collection %s", collection_name)
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        try:
            sparse_params = models.SparseVectorParams(modifier=models.Modifier.IDF)
        except Exception:
            sparse_params = models.SparseVectorParams()
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": sparse_params,
            },
        )
        logger.info("Created Qdrant collection %s with vector size %s", collection_name, vector_size)
    else:
        logger.info("Qdrant collection %s already exists", collection_name)

    try:
        from qdrant_client.models import SchemaKind
        schema_keyword = SchemaKind.KEYWORD
    except ImportError:
        schema_keyword = models.PayloadSchemaType.KEYWORD

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.asset_type",
            field_schema=schema_keyword,
        )
    except Exception as exc:
        logger.debug("Failed or skipped metadata.asset_type index: %s", exc)

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.asset_id",
            field_schema=schema_keyword,
        )
    except Exception as exc:
        logger.debug("Failed or skipped metadata.asset_id index: %s", exc)

    print("⚡ Hardware-accelerated payload lookup indexes established.")

    create_payload_indexes(client=client, collection_name=collection_name)


def create_payload_indexes(client: object, collection_name: str) -> None:
    """Create payload indexes used for common metadata filters."""

    from qdrant_client import models

    schema_map = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "bool": models.PayloadSchemaType.BOOL,
        "text": models.PayloadSchemaType.TEXT,
    }
    for field_name, schema_name in qdrant_payload_indexes().items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_map[schema_name],
            )
            logger.info("Payload index ready on %s: %s (%s)", collection_name, field_name, schema_name)
        except Exception as exc:
            logger.debug("Payload index %s may already exist on %s: %s", field_name, collection_name, exc)


if __name__ == "__main__":
    import argparse
    from embeddings.embedding_model import EmbeddingModelSettings

    parser = argparse.ArgumentParser(description="Create or recreate the Qdrant collection used by the RAG pipeline.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection before ingestion.")
    parser.add_argument("--collection", default=None, help="Override the configured Qdrant collection name.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("Creating Qdrant collection...")

    create_collection(
        vector_size=EmbeddingModelSettings().embedding_dimension,
        collection_name=args.collection,
        recreate=args.recreate,
    )

    print("Collection creation completed!")
