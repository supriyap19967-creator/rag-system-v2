# Qdrant Vector Database

The `vectordb/` package stores BGE-M3 embeddings from multimodal-enriched chunks in Qdrant for conversational RAG.

## Local Docker

```bash
docker compose up -d qdrant
```

Qdrant will be available at:

- HTTP: `http://localhost:6333`
- gRPC: `localhost:6334`

## Environment

```bash
QDRANT_COLLECTION=conversational_rag
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_PREFER_GRPC=false
QDRANT_TIMEOUT_SECONDS=30
```

For Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY`.

## Initialize And Ingest

```bash
python scripts/init_qdrant_collection.py --collection conversational_rag
python -m vectordb.example_usage Data/Pdf/example.pdf --collection conversational_rag --recreate
python -m vectordb.example_usage Data/csv/example.csv --collection conversational_rag
```

## Search Filters

The search layer supports simple metadata filters:

```python
filters = {
    "document_type": "pdf",
    "contains_chart": True,
    "source_file": "annual_report.pdf",
}
```

Range filters use:

```python
filters = {"page": {"gte": 10, "lte": 20}}
```

## Modules

- `qdrant_client_manager.py`: local, Docker, and cloud-ready clients.
- `create_collection.py`: BGE-M3 cosine collection and payload indexes.
- `ingest_vectors.py`: batch upsert with deterministic UUIDs.
- `search_vectors.py`: top-k semantic search with filters and score thresholds.
- `metadata_schema.py`: canonical PDF/CSV/table/chart/diagram payloads.
- `retrieval_pipeline.py`: conversational query embedding and retrieval context.
