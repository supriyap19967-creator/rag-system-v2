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

import os
import uuid
import json
import csv
import logging
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
VISUAL_JSONL_PATH = PROJECT_ROOT / "visual_chunks_output.jsonl"
csv_dir = PROJECT_ROOT / "Data" / "csv"
pdf_path = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"

# Import existing pipeline components to parse PDF text-only chunks
from ingest_data import parse_sources

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

def _stable_chunk_id(chunk_content: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_content)))

def main():
    logger.info("Starting complete Qdrant deployment for 2126 chunks...")

    # 1. Gather PDF text-only chunks (834 chunks)
    logger.info("Parsing PDF text-only chunks...")
    if not pdf_path.exists():
        logger.error(f"PDF file not found at {pdf_path}!")
        sys.exit(1)
        
    # Set environment to skip visuals during this parse to get text-only chunks
    os.environ["INGESTION_EXTRACT_FIGURES"] = "false"
    os.environ["INGESTION_USE_VISION"] = "false"
    
    pdf_records = parse_sources([pdf_path], enrich_pdf_visuals=False)
    text_chunks = []
    for record in pdf_records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        metadata = dict(record.get("metadata") or {})
        metadata["document_type"] = "pdf"
        chunk_id = metadata.get("chunk_id") or _stable_chunk_id(text)
        
        payload = {
            "text": text,
            "page_content": text,
            "source": record.get("source", pdf_path.name),
            "contains_chart": False,
            "contains_table": False,
            "contains_figure": False,
            "contains_image": False,
            "contains_csv": False,
            "document_type": "pdf",
            "metadata": metadata,
        }
        text_chunks.append({
            "text": text,
            "chunk_id": chunk_id,
            "payload": payload
        })
    logger.info(f"Loaded {len(text_chunks)} PDF text-only chunks.")

    # 2. Gather custom timeline-paired CSV chunks (513 chunks)
    logger.info("Generating custom timeline-paired CSV chunks...")
    metadata_map = {}
    for prefix in ["GDP", "CO2"]:
        meta_file = csv_dir / f"{prefix}2.csv"
        if not meta_file.exists():
            continue
        meta_rows = read_csv_as_dicts(meta_file, "Country Code")
        for r in meta_rows:
            code = r.get("Country Code", "").strip()
            if code:
                metadata_map[code] = {
                    "Region": r.get("Region", "").strip(),
                    "IncomeGroup": r.get("IncomeGroup", "").strip()
                }

    csv_chunks = []
    for prefix in ["GDP", "CO2"]:
        data_file = csv_dir / f"{prefix}1.csv"
        if not data_file.exists():
            continue
        data_rows = read_csv_as_dicts(data_file, "Country Name")
        
        for idx, row in enumerate(data_rows):
            country_name = row.get("Country Name", "").strip()
            country_code = row.get("Country Code", "").strip()
            indicator_name = row.get("Indicator Name", "").strip()
            
            meta = metadata_map.get(country_code, {"Region": "", "IncomeGroup": ""})
            region = meta.get("Region", "").strip()
            income_group = meta.get("IncomeGroup", "").strip()
            
            year_cols = sorted([k for k in row.keys() if k.isdigit() and len(k) == 4])
            
            historical_lines = []
            for y in year_cols:
                val = row.get(y, "")
                if is_valid_value(val):
                    historical_lines.append(f"- {y}: {val.strip()}")
            
            if not historical_lines:
                continue
                
            text_payload = f"Country: {country_name} (Code: {country_code})\n"
            text_payload += f"Region: {region} | Income Group: {income_group}\n"
            text_payload += f"Indicator: {indicator_name}\n"
            text_payload += "Historical Data:\n"
            text_payload += "\n".join(historical_lines)
            
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text_payload))
            payload = {
                "text": text_payload,
                "page_content": text_payload,
                "source": data_file.name,
                "contains_chart": False,
                "contains_table": True,
                "contains_figure": False,
                "contains_image": False,
                "contains_csv": True,
                "document_type": "csv",
                "metadata": {
                    "chunk_id": chunk_id,
                    "document_type": "csv",
                    "source_file": data_file.name,
                    "source_path": str(data_file),
                    "row_id": idx,
                    "contains_csv": True,
                    "contains_table": True,
                    "contains_chart": False,
                    "contains_figure": False,
                    "contains_image": False,
                },
            }
            csv_chunks.append({
                "text": text_payload,
                "chunk_id": chunk_id,
                "payload": payload
            })
    logger.info(f"Loaded {len(csv_chunks)} custom CSV chunks.")

    # 3. Gather visual chunks from visual_chunks_output.jsonl (779 chunks)
    logger.info("Loading visual chunks from cache...")
    if not VISUAL_JSONL_PATH.exists():
        logger.error(f"Visual chunks JSONL file not found at {VISUAL_JSONL_PATH}!")
        sys.exit(1)
        
    visual_chunks = []
    with open(VISUAL_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            metadata = dict(item.get("metadata") or {})
            
            # Ensure proper document type mapping
            metadata["document_type"] = "pdf_visual"
            chunk_id = str(metadata.get("chunk_id") or _stable_chunk_id(text))
            
            payload = {
                "text": text,
                "page_content": text,
                "source": str(metadata.get("source_file") or metadata.get("source") or pdf_path.name),
                "image_path": metadata.get("image_path") or metadata.get("figure_image_path"),
                "contains_chart": bool(metadata.get("contains_chart") or "[visual element" in text.lower()),
                "contains_table": bool(metadata.get("contains_table") or "|--" in text),
                "contains_figure": bool(metadata.get("contains_figure") or "figure" in text.lower()),
                "contains_image": bool(metadata.get("contains_image") or "image" in text.lower()),
                "contains_csv": bool(metadata.get("contains_csv")),
                "document_type": "pdf_visual",
                "metadata": metadata,
            }
            
            # Transfer asset specific metadata mappings same
            for key in ["row_id", "columns"]:
                if metadata.get(key) not in ("", None, [], {}):
                    payload[key] = metadata[key]
                    
            visual_chunks.append({
                "text": text,
                "chunk_id": chunk_id,
                "payload": payload
            })
    logger.info(f"Loaded {len(visual_chunks)} visual chunks from cache.")

    # 4. Consolidate and check count
    all_chunks = text_chunks + csv_chunks + visual_chunks
    total_count = len(all_chunks)
    logger.info(f"Consolidated total chunks gathered: {total_count} (Expected: 2126)")

    # 5. Setup Qdrant Client and recreate collection
    logger.info("Recreating Qdrant collection with vector size 384...")
    client = QdrantClient(path=str(QDRANT_PATH))
    
    if client.collection_exists(COLLECTION_NAME):
        logger.info(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        
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
    logger.info("Collection created successfully.")

    # 6. Load SentenceTransformer and generate embeddings
    logger.info("Loading SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [item["text"] for item in all_chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks locally...")
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    logger.info("Embeddings generated successfully.")

    # 7. Construct points and upsert to Qdrant
    logger.info("Constructing Qdrant points...")
    points = []
    for idx, item in enumerate(all_chunks):
        chunk_id = item["chunk_id"]
        payload = item["payload"]
        dense_vector = [float(val) for val in embeddings[idx]]
        
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
        
    # Upsert in batches
    batch_size = 64
    uploaded = 0
    total_points = len(points)
    logger.info(f"Uploading {total_points} points to Qdrant in batches of {batch_size}...")
    for start in range(0, total_points, batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded += len(batch)
        logger.info(f"Upserted {uploaded}/{total_points} points.")

    # 8. Retrieve final counts breakdown from Qdrant
    logger.info("Verifying final database counts...")
    final_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    
    offset = None
    counts = {"csv": 0, "text only": 0, "visual": 0, "unknown": 0}
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata", {})
            doc_type = payload.get("document_type") or metadata.get("document_type") or "unknown"
            
            if doc_type == "csv":
                counts["csv"] += 1
            elif doc_type == "pdf":
                counts["text only"] += 1
            elif doc_type == "pdf_visual":
                counts["visual"] += 1
            else:
                counts["unknown"] += 1
                
        if not next_offset:
            break
        offset = next_offset

    client.close()

    # Print final summary
    sep = "=" * 65
    print()
    print(sep)
    print("  FINAL QDRANT DEPLOYMENT COMPLETE")
    print(sep)
    print()
    print(f"  CSV Chunks (Paired):.................... {counts['csv']:>6}")
    print(f"  Text-Only Chunks (PDF):................. {counts['text only']:>6}")
    print(f"  Visual Chunks (PDF Visual):............. {counts['visual']:>6}")
    print(f"  Unknown Chunks:......................... {counts['unknown']:>6}")
    print(f"  Total Chunks in Qdrant:................. {final_count:>6}")
    print()
    print(sep)
    print()

if __name__ == "__main__":
    main()
