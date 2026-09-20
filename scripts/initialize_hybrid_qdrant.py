from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "rag_master_v2"
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"

def get_qdrant_client_local() -> QdrantClient:
    """Initialize local Qdrant client."""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=qdrant_host, port=qdrant_port)

def initialize_hybrid_collection(recreate_forced: bool = False) -> None:
    """
    Creates/resets the hybrid Qdrant collection with:
    - 'text-dense': 384-dimensional cosine similarity space.
    - 'text-sparse': Qdrant's native sparse vector parameters.
    """
    client = get_qdrant_client_local()
    
    logger.info("Initializing embedding generators...")
    # Initialize Embedding generators
    dense_embedder = TextEmbedding(DENSE_MODEL_NAME)
    sparse_embedder = SparseTextEmbedding(SPARSE_MODEL_NAME)
    logger.info(f"Loaded Dense: {DENSE_MODEL_NAME} | Sparse: {SPARSE_MODEL_NAME}")
    
    exists = client.collection_exists(collection_name=COLLECTION_NAME)
    recreate_needed = recreate_forced
    
    if exists and not recreate_needed:
        # Check if sparse vector support exists
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        sparse_config = getattr(collection_info.config, "sparse_vectors_config", None)
        if not sparse_config or "text-sparse" not in sparse_config:
            logger.warning(f"Collection '{COLLECTION_NAME}' exists but lacks 'text-sparse' config. Recreating...")
            recreate_needed = True
            
    if exists and recreate_needed:
        logger.info(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        exists = False
        
    if not exists:
        logger.info(f"Creating collection '{COLLECTION_NAME}' with hybrid dense/sparse vector configuration...")
        
        # Configure native sparse params safely
        try:
            sparse_params = models.SparseVectorParams(modifier=models.Modifier.IDF)
        except Exception:
            sparse_params = models.SparseVectorParams()
            
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=384,  # Size of BAAI/bge-small-en-v1.5
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "text-sparse": sparse_params
            }
        )
        logger.info(f"Collection '{COLLECTION_NAME}' successfully created.")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists with correct sparse support. Skipping creation.")

    # Establish hardware-accelerated payload lookup indexes
    for field in ["metadata.asset_type", "metadata.asset_id", "metadata.source_file"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created payload index on: '{field}'")
        except Exception as e:
            logger.debug(f"Payload index creation skipped for '{field}': {e}")

if __name__ == "__main__":
    initialize_hybrid_collection(recreate_forced=True)
