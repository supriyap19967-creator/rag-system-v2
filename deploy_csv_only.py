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
import csv
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Add project root to path to resolve custom modules correctly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "conversational_rag"
QDRANT_PATH = PROJECT_ROOT / "qdrant_db"

def is_valid_value(val):
    if val is None:
        return False
    v = val.strip().lower()
    if v in ("", "nan", "null", "none"):
        return False
    return True

def read_csv_as_dicts(filepath, header_first_col):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if row and row[0].strip().strip('"') == header_first_col:
                header = [col.strip().strip('"') for col in row]
                break
        if not header:
            return []
        
        rows = []
        for r in csv.reader(f):
            if not r:
                continue
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            else:
                r = r[:len(header)]
            rows.append(dict(zip(header, r)))
        return rows

def main():
    csv_dir = PROJECT_ROOT / "Data" / "csv"
    logger.info("Starting CSV-only deployment with updated pairing and structural logic...")

    # DRY_RUN is set to True to chunk and stop, keeping Qdrant data as it is.
    # Set to False to perform actual database update (delete CSV chunks and upsert new ones).
    DRY_RUN = False

    # Load metadata maps from "2" files (GDP2.csv, CO22.csv)
    metadata_map = {}
    for prefix in ["GDP", "CO2"]:
        meta_file = csv_dir / f"{prefix}2.csv"
        if not meta_file.exists():
            logger.warning(f"Metadata file not found: {meta_file}")
            continue
        logger.info(f"Reading metadata from {meta_file.name}...")
        meta_rows = read_csv_as_dicts(meta_file, "Country Code")
        for r in meta_rows:
            code = r.get("Country Code", "").strip()
            if code:
                metadata_map[code] = {
                    "Region": r.get("Region", "").strip(),
                    "IncomeGroup": r.get("IncomeGroup", "").strip()
                }

    records = []
    # Process "1" data files (GDP1.csv, CO21.csv)
    for prefix in ["GDP", "CO2"]:
        data_file = csv_dir / f"{prefix}1.csv"
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            continue
        logger.info(f"Reading timeline metrics from {data_file.name}...")
        data_rows = read_csv_as_dicts(data_file, "Country Name")
        
        for idx, row in enumerate(data_rows):
            country_name = row.get("Country Name", "").strip()
            country_code = row.get("Country Code", "").strip()
            indicator_name = row.get("Indicator Name", "").strip()
            
            # Lookup metadata
            meta = metadata_map.get(country_code, {"Region": "", "IncomeGroup": ""})
            region = meta.get("Region", "").strip()
            income_group = meta.get("IncomeGroup", "").strip()
            
            # Extract years (1960 to recent years)
            year_cols = sorted([k for k in row.keys() if k.isdigit() and len(k) == 4])
            
            historical_lines = []
            for y in year_cols:
                val = row.get(y, "")
                if is_valid_value(val):
                    historical_lines.append(f"- {y}: {val.strip()}")
            
            if not historical_lines:
                continue
                
            # Format text payload
            text_payload = f"Country: {country_name} (Code: {country_code})\n"
            text_payload += f"Region: {region} | Income Group: {income_group}\n"
            text_payload += f"Indicator: {indicator_name}\n"
            text_payload += "Historical Data:\n"
            text_payload += "\n".join(historical_lines)
            
            records.append({
                "text": text_payload,
                "source": data_file.name,
                "metadata": {
                    "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, text_payload)),
                    "document_type": "csv",
                    "source_file": data_file.name,
                    "source_path": str(data_file),
                    "row_id": idx,
                    "contains_csv": True,
                    "contains_table": True,
                    "contains_chart": False,
                    "contains_figure": False,
                    "contains_image": False,
                }
            })
            
    total_records = len(records)
    logger.info(f"Successfully processed {total_records} CSV chunks in total.")

    if DRY_RUN:
        logger.info("[DRY RUN] Bypassing Qdrant deletion and ingestion to keep database data as it is.")
        print(f"\nCSV-only Ingestion Completed (DRY RUN).")
        print(f"Processed chunks: {total_records}")
        return

    # 2. Setup Qdrant Client
    client = QdrantClient(url="http://localhost:6333")
    
    # Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}' with vector size 384...")
        try:
            sparse_params = models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
                modifier=models.Modifier.IDF,
            )
        except Exception:
            sparse_params = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": sparse_params},
        )
    else:
        # Delete only CSV-only chunks from collection
        logger.info(f"Deleting existing CSV-only chunks from collection '{COLLECTION_NAME}'...")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                should=[
                    models.FieldCondition(key="document_type", match=models.MatchValue(value="csv")),
                    models.FieldCondition(key="metadata.document_type", match=models.MatchValue(value="csv"))
                ]
            )
        )

    # 3. Load Local SentenceTransformer Model
    logger.info("Loading local SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Local SentenceTransformer model loaded successfully.")

    # 4. Generate Embeddings for all chunks
    texts = [record["text"] for record in records]
    logger.info(f"Generating embeddings for {len(texts)} chunks locally...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    logger.info("Embedding generation completed.")

    # 5. Construct Qdrant points
    points = []
    for i, record in enumerate(records):
        text = record["text"]
        metadata = record["metadata"]
        chunk_id = metadata["chunk_id"]

        payload = {
            "text": text,
            "page_content": text,
            "source": record["source"],
            "contains_chart": False,
            "contains_table": True,
            "contains_figure": False,
            "contains_image": False,
            "contains_csv": True,
            "document_type": "csv",
            "metadata": metadata,
        }

        dense_vector = [float(val) for val in embeddings[i]]

        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                vector={
                    "dense": dense_vector,
                    "sparse": models.SparseVector(indices=[], values=[]),
                },
                payload=payload,
            )
        )

    # 6. Upsert points in batches
    batch_size = 64
    uploaded = 0
    total_points = len(points)
    for start in range(0, total_points, batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded += len(batch)
        logger.info(f"Upserted {uploaded}/{total_points} points.")

    # Retrieve final exact count
    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    client.close()

    logger.info("CSV-only deployment completed. Points in collection '%s': %d", COLLECTION_NAME, count)
    print(f"\nCSV-only Ingestion Completed.")
    print(f"Upserted Points: {uploaded}")
    print(f"Qdrant Exact Count: {count}")

if __name__ == "__main__":
    main()
