import sys
import os
import uuid
import csv
import logging
from pathlib import Path
from qdrant_client import QdrantClient, models

# Setup paths and logger
PROJECT_ROOT = Path(__file__).resolve().parent
QDRANT_PATH = PROJECT_ROOT / "qdrant_db"
csv_dir = PROJECT_ROOT / "Data" / "csv"
COLLECTION_NAME = "conversational_rag"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    # 1. Connect to Qdrant and delete all CSV chunks
    if not QDRANT_PATH.exists():
        logger.error(f"Qdrant database not found at {QDRANT_PATH}")
        sys.exit(1)
        
    client = QdrantClient(path=str(QDRANT_PATH))
    if not client.collection_exists(COLLECTION_NAME):
        logger.error(f"Collection '{COLLECTION_NAME}' does not exist.")
        sys.exit(1)
        
    logger.info("Deleting existing 1062 raw CSV chunks from Qdrant...")
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            should=[
                models.FieldCondition(key="document_type", match=models.MatchValue(value="csv")),
                models.FieldCondition(key="metadata.document_type", match=models.MatchValue(value="csv"))
            ]
        )
    )
    
    # 2. Get current counts in Qdrant (showing text-only and visual chunks are kept as-is)
    total_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    
    # Scroll to get document type breakdown
    offset = None
    counts_in_qdrant = {"csv": 0, "text only": 0, "visual": 0, "unknown": 0}
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
                counts_in_qdrant["csv"] += 1
            elif doc_type == "pdf":
                counts_in_qdrant["text only"] += 1
            elif doc_type == "pdf_visual":
                counts_in_qdrant["visual"] += 1
            else:
                counts_in_qdrant["unknown"] += 1
                
        if not next_offset:
            break
        offset = next_offset
        
    client.close()

    # 3. Generate new CSV chunks locally (without adding to Qdrant)
    logger.info("Generating new high-quality timeline-paired CSV chunks locally...")
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

    new_csv_records = []
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
            
            new_csv_records.append({
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
            
    new_csv_count = len(new_csv_records)

    # 4. Print consolidated report
    sep = "=" * 65
    print()
    print(sep)
    print("  CONSOLIDATED CHUNK COUNT & QDRANT STATUS REPORT")
    print(sep)
    print()
    print("  QDRANT DATABASE STATUS (AFTER DELETING RAW CSV CHUNKS):")
    print(f"    CSV Chunks in Qdrant:................. {counts_in_qdrant['csv']:>6}")
    print(f"    Text-Only Chunks in Qdrant:........... {counts_in_qdrant['text only']:>6}")
    print(f"    Visual Chunks in Qdrant:.............. {counts_in_qdrant['visual']:>6}")
    print(f"    Total Points in Qdrant:............... {total_count:>6}")
    print()
    print("  NEW LOCALLY-GENERATED CHUNKS (NOT ADDED TO QDRANT YET):")
    print(f"    New Paired CSV Chunks:................ {new_csv_count:>6}")
    print()
    print("  CONSOLIDATED PLAN TOTALS (KEEPING EXISTING + NEW CSV):")
    total_plan = counts_in_qdrant['text only'] + counts_in_qdrant['visual'] + new_csv_count
    print(f"    Text-Only Chunks (Existing):.......... {counts_in_qdrant['text only']:>6}")
    print(f"    Visual Chunks (Existing):............. {counts_in_qdrant['visual']:>6}")
    print(f"    New Paired CSV Chunks (Generated):.... {new_csv_count:>6}")
    print(f"    Consolidated Total Chunk Plan:........ {total_plan:>6}")
    print()
    print(sep)
    print()

if __name__ == "__main__":
    main()
