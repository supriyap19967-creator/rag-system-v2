import sys
import types
import datasets

# Mock sentence_transformers trainer, training_args, cross_encoder, and sparse_encoder to bypass Trainer imports
sys.modules['sentence_transformers.trainer'] = types.ModuleType('sentence_transformers.trainer')
sys.modules['sentence_transformers.trainer'].SentenceTransformerTrainer = None

sys.modules['sentence_transformers.training_args'] = types.ModuleType('sentence_transformers.training_args')
sys.modules['sentence_transformers.training_args'].SentenceTransformerTrainingArguments = None
sys.modules['sentence_transformers.training_args'].BatchSamplers = None
sys.modules['sentence_transformers.training_args'].MultiDatasetBatchSamplers = None

sys.modules['sentence_transformers.sparse_encoder'] = types.ModuleType('sentence_transformers.sparse_encoder')
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoder = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderModelCardData = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainer = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainingArguments = None

sys.modules['sentence_transformers.cross_encoder'] = types.ModuleType('sentence_transformers.cross_encoder')
sys.modules['sentence_transformers.cross_encoder'].CrossEncoder = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderModelCardData = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainer = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainingArguments = None

import logging
import os
import uuid
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Add project root to path to resolve custom modules correctly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from ingestion.pipeline import MultimodalIngestionPipeline
from ingestion.config import IngestionSettings

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "conversational_rag"
QDRANT_PATH = PROJECT_ROOT / "qdrant_db"

def _stable_chunk_id(chunk_content: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_content)))

def main():
    pdf_path = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"
    logger.info(f"Starting text-only PDF deployment to Qdrant for {pdf_path.name}...")

    # 1. Initialize pipeline with figure extraction and vision disabled (text-only)
    settings = IngestionSettings(
        extract_figures=False,
        use_vision=False
    )
    pipeline = MultimodalIngestionPipeline(settings)

    # 2. Ingest the PDF to get text chunks
    logger.info("Ingesting PDF text chunks...")
    result = pipeline.ingest_sync(pdf_path)
    chunks = result.chunks
    logger.info(f"Successfully parsed {len(chunks)} PDF text chunks.")

    # 3. Setup Qdrant Client (without recreating the collection)
    client = QdrantClient(path=str(QDRANT_PATH))
    if not client.collection_exists(COLLECTION_NAME):
        logger.error(f"Collection '{COLLECTION_NAME}' does not exist! Please run CSV deployment first.")
        sys.exit(1)

    # 4. Load Local SentenceTransformer Model
    logger.info("Loading local SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Local SentenceTransformer model loaded successfully.")

    # 5. Generate Embeddings for all text chunks
    texts = [str(chunk.text).strip() for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks locally...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    logger.info("Embedding generation completed.")

    # 6. Construct Qdrant points
    points = []
    for i, chunk in enumerate(chunks):
        text = str(chunk.text).strip()
        metadata = dict(chunk.metadata or {})
        chunk_id = str(metadata.get("chunk_id") or _stable_chunk_id(text))

        payload = {
            "text": text,
            "page_content": text,
            "source": str(metadata.get("source_file") or pdf_path.name),
            "image_path": metadata.get("image_path"),
            "contains_chart": bool(metadata.get("contains_chart")),
            "contains_table": bool(metadata.get("contains_table")),
            "contains_figure": bool(metadata.get("contains_figure")),
            "contains_image": bool(metadata.get("contains_image")),
            "contains_csv": bool(metadata.get("contains_csv")),
            "metadata": metadata,
        }
        # Keep any asset specific metadata mappings same
        for key in ["row_id", "columns"]:
            if metadata.get(key) not in ("", None, [], {}):
                payload[key] = metadata[key]

        dense_vector = [float(val) for val in embeddings[i]]

        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector={
                    "dense": dense_vector,
                    "sparse": models.SparseVector(indices=[], values=[]),
                },
                payload=payload,
            )
        )

    # 7. Upsert points in batches (appending to the existing collection)
    batch_size = 64
    uploaded = 0
    total_points = len(points)
    for start in range(0, total_points, batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded += len(batch)
        logger.info(f"Upserted {uploaded}/{total_points} points.")

    # Retrieve final exact count of all points (including CSV)
    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    client.close()

    logger.info("PDF text-only deployment completed. Total points in collection '%s': %d", COLLECTION_NAME, count)
    print(f"\nPDF Text-only Ingestion Completed.")
    print(f"Upserted PDF Points: {uploaded}")
    print(f"Qdrant Exact Count (Total): {count}")

if __name__ == "__main__":
    main()
