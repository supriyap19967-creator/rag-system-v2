from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from embeddings.embed_chunks import ChunkEmbedder
from ingestion.pipeline import MultimodalIngestionPipeline
from vectordb.create_collection import create_collection
from vectordb.ingest_vectors import QdrantVectorIngester
from vectordb.retrieval_pipeline import ConversationalRetrievalPipeline


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest multimodal content into Qdrant and run retrieval.")
    parser.add_argument("source", type=Path, help="PDF or CSV source file.")
    parser.add_argument("--collection", default="conversational_rag")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--query", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    ingestion_result = await MultimodalIngestionPipeline().ingest(args.source)
    embedded_chunks = await ChunkEmbedder().aembed_chunks(ingestion_result.chunks)

    vector_size = len(embedded_chunks[0].embedding) if embedded_chunks else 1024
    create_collection(vector_size=vector_size, collection_name=args.collection, recreate=args.recreate)
    uploaded = QdrantVectorIngester(collection_name=args.collection).ingest(embedded_chunks)
    print(f"Uploaded {uploaded} chunks to Qdrant collection {args.collection}.")

    if args.query:
        context = ConversationalRetrievalPipeline().retrieve(args.query, top_k=5)
        for result in context.results:
            print(f"\nscore={result.score:.4f} source={result.metadata.get('source_file')} page={result.metadata.get('page')}")
            print(result.text[:700])


if __name__ == "__main__":
    asyncio.run(main())
