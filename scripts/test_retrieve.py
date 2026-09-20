import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)

from query_rag import retrieve_context, build_qdrant_client

def test_retrieve():
    qdrant = build_qdrant_client()
    query = "explain figure 7.5"
    print(f"Retrieving context for: {query}")
    results = retrieve_context(qdrant, query)
    print(f"\nRetrieved {len(results)} chunks:")
    for idx, r in enumerate(results):
        print(f"\n[{idx}] Score: {r['score']}")
        print(f"Text preview: {r['text'][:200]}...")
        print(f"Metadata: {r['metadata']}")

if __name__ == "__main__":
    test_retrieve()
