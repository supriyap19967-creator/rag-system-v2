from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from scripts.initialize_hybrid_qdrant import COLLECTION_NAME, get_qdrant_client_local, initialize_hybrid_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_verification() -> None:
    # 1. Initialize Hybrid Qdrant Collection
    logger.info("Initializing Hybrid Collection...")
    initialize_hybrid_collection(recreate_forced=True)
    
    client = get_qdrant_client_local()
    
    # 2. Initialize Embeddings
    logger.info("Generating sample embeddings...")
    dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    
    sample_text = (
        "Figure 4.1: Global economic trend showing GDP growth by income levels. "
        "The data is structured in assets/extracted_tables/page_206_Table_4.1.csv and "
        "its chart crop is located at assets/extracted_images/page_206_Figure_4.1.png."
    )
    
    dense_vector = list(dense_model.embed([sample_text]))[0].tolist()
    sparse_vector = list(sparse_model.embed([sample_text]))[0]
    
    # 3. Create Qdrant Sparse Vector Structure
    qdrant_sparse_vector = models.SparseVector(
        indices=sparse_vector.indices.tolist(),
        values=sparse_vector.values.tolist()
    )
    
    # 4. Multimodal Compatibility Payload
    payload = {
        "content": sample_text,
        "metadata": {
            "source_file": "World Development Report 2025.pdf",
            "page_number": 206,
            "asset_type": "figure",
            "asset_id": "Figure_4.1",
            "image_path": "assets/extracted_images/page_206_Figure_4.1.png",
            "table_path": "assets/extracted_tables/page_206_Table_4.1.csv",
            "bounding_boxes": [[0.1, 0.2, 0.9, 0.8]]
        }
    }
    
    # 5. Upsert Point
    logger.info("Upserting verification point into hybrid namespace...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=9999,
                vector={
                    "text-dense": dense_vector,
                    "text-sparse": qdrant_sparse_vector
                },
                payload=payload
            )
        ]
    )
    
    # 6. Verify and Retrieve
    logger.info("Performing Search retrieval using hybrid query namespace...")
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(
            name="text-dense",
            vector=dense_vector
        ),
        limit=1,
        with_payload=True
    )
    
    if search_results:
        retrieved = search_results[0]
        logger.info("Verification Success! Retrieved Point Details:")
        logger.info(f"  Point ID: {retrieved.id}")
        logger.info(f"  Asset ID: {retrieved.payload.get('metadata', {}).get('asset_id')}")
        logger.info(f"  Image Path: {retrieved.payload.get('metadata', {}).get('image_path')}")
        logger.info(f"  Table Path: {retrieved.payload.get('metadata', {}).get('table_path')}")
        logger.info(f"  Bounding Box: {retrieved.payload.get('metadata', {}).get('bounding_boxes')}")
    else:
        raise RuntimeError("Failed to retrieve the verification point.")

if __name__ == "__main__":
    run_verification()
