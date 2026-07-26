import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def audit_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Error: Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    namespace = os.getenv("PINECONE_NAMESPACE", "bge_small_v1")
    
    print(f"Auditing Pinecone Index: {index_name}, Namespace: {namespace}")
    
    dummy_vector = [0.0] * 384

    # Query for CSV count
    csv_stats = index.query(
        vector=dummy_vector,
        namespace=namespace,
        top_k=5,
        filter={"source_type": {"$eq": "csv"}},
        include_metadata=True
    )
    
    # PDF count
    pdf_stats = index.query(
        vector=dummy_vector,
        namespace=namespace,
        top_k=5,
        filter={"source_type": {"$eq": "pdf"}},
        include_metadata=True
    )
    
    print("\n--- CSV Sample Metadata (using filter 'source_type'=='csv') ---")
    if csv_stats.matches:
        for m in csv_stats.matches:
            print(f"ID: {m.id}, Score: {m.score}, Metadata Keys: {list(m.metadata.keys())}")
            print(f"Sample Content: {m.metadata.get('original_text')[:100]}...")
            print(f"Source Type: {m.metadata.get('source_type')}")
    else:
        print("No CSV chunks found with 'source'=='csv'.")

    print("\n--- PDF Sample Metadata (using filter 'source_type'=='pdf') ---")
    if pdf_stats.matches:
        for m in pdf_stats.matches:
            print(f"ID: {m.id}, Score: {m.score}, Metadata Keys: {list(m.metadata.keys())}")
            print(f"Sample Content: {m.metadata.get('original_text')[:100]}...")
            print(f"Source Type: {m.metadata.get('source_type')}")
    else:
        print("No PDF chunks found with 'source'=='pdf'.")

    # Check for 'source_type' just in case
    print("\n--- Checking for 'source_type' key ---")
    res_type = index.query(vector=dummy_vector, namespace=namespace, top_k=1, filter={"source_type": {"$exists": True}})
    if res_type.matches:
        print(f"Found matches with 'source_type' key. Sample: {res_type.matches[0].metadata}")
    else:
        print("No matches with 'source_type' key.")

    # Exhaustive case check
    print("\n--- Case Sensitivity Check ---")
    for key in ["source", "source_type"]:
        for val in ["CSV", "csv", "PDF", "pdf"]:
            res = index.query(vector=dummy_vector, namespace=namespace, top_k=1, filter={key: {"$eq": val}})
            if res.matches:
                print(f"Match found for {key}='{val}'")

if __name__ == "__main__": audit_pinecone()
