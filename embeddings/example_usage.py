from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from embeddings.embed_chunks import ChunkEmbedder
from embeddings.vector_store import FaissVectorStore
from ingestion.pipeline import MultimodalIngestionPipeline


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, embed with BGE-M3, and persist a FAISS vector store.")
    parser.add_argument("source", type=Path, help="Path to an enriched-source PDF or CSV.")
    parser.add_argument("--persist-dir", type=Path, default=Path("Data/vectorstores/bge_m3_faiss"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    ingestion_result = await MultimodalIngestionPipeline().ingest(args.source)
    embedded_chunks = await ChunkEmbedder().aembed_chunks(ingestion_result.chunks)

    store = FaissVectorStore(args.persist_dir)
    store.build(embedded_chunks)
    store.save()

    print(json.dumps({"store": store.to_dict(), "embedded_chunks": len(embedded_chunks)}, indent=2))
    if args.query:
        for result in store.search(args.query, top_k=args.top_k):
            print(json.dumps({"score": result.score, "metadata": result.metadata, "text": result.text[:500]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
