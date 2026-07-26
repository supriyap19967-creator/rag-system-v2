# Multimodal Ingestion Pipeline

This project includes a production-style ingestion package at `ingestion/` for building enriched, embedding-ready content for conversational RAG.

## Flow

1. PDFs are parsed with Docling into structured markdown, preserving headings, lists, sections, and tables.
2. PDF visual blocks are detected with unstructured.io and extracted to `assets/extracted_images` by default.
3. Extracted charts, diagrams, and figures are captioned with Gemini 2.5 Flash Vision using a structured factual prompt.
4. Vision descriptions are appended back into the markdown inside `[CHART DESCRIPTION]` blocks.
5. CSV files are routed with unstructured.io, then converted from raw rows into natural-language retrieval sentences.
6. Enriched markdown is split with a markdown-aware recursive chunker that preserves tables and chart blocks.

## Environment

```bash
INGESTION_FIGURE_OUTPUT_DIR=assets/extracted_images
INGESTION_CHUNK_SIZE=1200
INGESTION_CHUNK_OVERLAP=180
INGESTION_USE_VISION=true
INGESTION_MAX_CONCURRENT_VISION_TASKS=2
GEMINI_API_KEYS=key_1,key_2,key_3
GEMINI_MODEL_NAME=gemini-2.0-flash
CACHE_DIR=data_cache/visual_captions
MAX_CONCURRENT_REQUESTS=2
```

## Usage

```bash
python -m ingestion.example_usage Data/Pdf/example.pdf --out Data/processed
python -m ingestion.example_usage Data/csv/example.csv --out Data/processed
```

The command writes:

- `*.enriched.md`: clean semantic markdown/text.
- `*.metadata.json`: source, page, image, and chart metadata.
- `*.chunks.jsonl`: embedding-ready chunks with retrieval metadata.

## Main Modules

- `ingestion.parse_pdf.DoclingPdfParser`
- `ingestion.parse_csv.CsvSemanticParser`
- `ingestion.detect_figures.FigureDetector`
- `ingestion.gemini_vision_caption.GeminiVisionCaptioner`
- `ingestion.merge_content.ContentMerger`
- `ingestion.chunking.MarkdownChunker`
- `ingestion.pipeline.MultimodalIngestionPipeline`
