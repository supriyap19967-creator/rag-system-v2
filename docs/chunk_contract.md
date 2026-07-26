# Chunk Contract

This project uses LlamaIndex as the unified ingestion layer. Every indexed item is a chunk with readable text plus metadata. Exact numeric answers must be validated from structured metadata, not inferred from nearest vector matches.

## Global Metadata

Every chunk should keep these fields when available:

- `source`
- `source_files`
- `source_type`
- `content_type`
- `page`
- `source_page`

CSV chunks also keep:

- `country_name`
- `country_iso3`
- `country_code`
- `year`
- `indicator`
- `indicator_code`
- `metric_family`
- `value`
- `row_index`

Visual chunks also keep:

- `visual_type`
- `figure_id`
- `caption`
- `image_path`
- `image_local_path`
- `crop_quality`
- `crop_quality_score`
- `crop_rejected_reason`

Metadata text fields should stay concise. Long fields are capped during ingestion so metadata does not overwhelm the embedding text.

## CSV Policy

CSV chunks are row-level value chunks.

Text template:

```text
In 2022, GDP (current US$) for India (IND) was 3346107287730.93.
```

Rules:

- Keep one complete country/year/indicator/value per chunk.
- Never split country, year, indicator, and value across chunks.
- Never round numeric values during ingestion.
- Store exact value as metadata.
- Normalize country code, for example `India -> IND` and `US / USA / United States -> USA`.
- Do not mix unrelated countries, years, or indicators in one chunk.
- Exact numeric questions use structured validation, not nearest vector search.
- If the exact country/year/metric match is missing, answer with insufficient data.

## PDF Policy

PDF text chunks are semantic paragraph chunks.

Recommended settings:

- `PDF_CHUNK_SIZE`: 700-1100 characters.
- `PDF_CHUNK_OVERLAP`: 80-150 characters.
- Minimum paragraph length: 40-60 characters.

Rules:

- Prefer complete paragraphs.
- Drop obvious page numbers, references, bibliography entries, and noisy fragments.
- Preserve page metadata for citations.

## Visual Policy

Visual chunks are one chunk per extracted visual asset.

Text should include:

- Clean caption.
- Visual type.
- Figure/table ID.
- Minimal nearby context only when useful.

Rules:

- Do not index body paragraphs as visual chunks.
- Remove repeated OCR lines and source-note noise.
- Keep `image_path`, `image_local_path`, `figure_id`, `page`, and `visual_type`.
- Exact figure/table requests must prefer the complete visual asset.
