# BGE-M3 Embedding Pipeline

The `embeddings/` package adds a production-style BGE-M3 dense retrieval layer for multimodal-enriched conversational RAG.

## Flow

1. `ingestion.MultimodalIngestionPipeline` emits enriched chunks containing original text, table summaries, chart descriptions, diagram explanations, and CSV semantic sentences.
2. `embeddings.ChunkEmbedder` batches those chunks through BGE-M3 and preserves enrichment metadata.
3. `embeddings.FaissVectorStore` stores normalized dense vectors plus text and metadata for conversational retrieval.

## Environment

```bash
BGE_M3_MODEL=BAAI/bge-m3
BGE_M3_DEVICE=cpu
BGE_M3_BATCH_SIZE=16
BGE_M3_MAX_SEQUENCE_LENGTH=8192
BGE_M3_EMBEDDING_DIMENSION=1024
BGE_M3_NORMALIZE=true
BGE_M3_BACKEND=flagembedding
BGE_M3_CACHE_FOLDER=hf_cache
```

`FlagEmbedding` is preferred because BGE-M3 can later support hybrid retrieval with sparse and ColBERT vectors. The implementation falls back to `sentence-transformers` for dense retrieval if needed.

## Usage

```bash
python -m embeddings.example_usage Data/Pdf/example.pdf --persist-dir Data/vectorstores/bge_m3_faiss
python -m embeddings.example_usage Data/csv/example.csv --query "Which quarter had the strongest revenue growth?"
```

## Metadata

Each embedded chunk includes fields such as:

- `source`
- `page`
- `chunk_id`
- `contains_chart`
- `contains_table`
- `contains_csv_semantic_sentence`
- `contains_diagram`
- `embedding_model`
- `embedding_backend`
- `embedding_dimension`
