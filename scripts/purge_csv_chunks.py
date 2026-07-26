import os
from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "conversational_rag"

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} does not exist.")
        return
        
    total_count_before = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Total points before deletion in '{COLLECTION_NAME}': {total_count_before}")
    
    # Let's see some document types before deletion
    limit = 100
    offset = None
    doc_types_before = {}
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata", {})
            doc_type = payload.get("document_type") or metadata.get("document_type") or "unknown"
            doc_types_before[doc_type] = doc_types_before.get(doc_type, 0) + 1
            
        if not next_offset:
            break
        offset = next_offset
    
    print("\nDocument type breakdown before deletion:")
    for dt, count in doc_types_before.items():
        print(f"  {dt}: {count}")
        
    print("\nDeleting CSV-only chunks from Qdrant...")
    result = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            should=[
                models.FieldCondition(key="document_type", match=models.MatchValue(value="csv")),
                models.FieldCondition(key="metadata.document_type", match=models.MatchValue(value="csv"))
            ]
        )
    )
    print(f"Delete operation returned: {result}")
    
    total_count_after = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Total points after deletion in '{COLLECTION_NAME}': {total_count_after}")
    
    # Scroll and print breakdown after deletion
    offset = None
    doc_types_after = {}
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata", {})
            doc_type = payload.get("document_type") or metadata.get("document_type") or "unknown"
            doc_types_after[doc_type] = doc_types_after.get(doc_type, 0) + 1
            
        if not next_offset:
            break
        offset = next_offset
        
    print("\nRemaining Document type breakdown:")
    for dt, count in doc_types_after.items():
        print(f"  {dt}: {count}")
        
    client.close()

if __name__ == "__main__":
    main()
