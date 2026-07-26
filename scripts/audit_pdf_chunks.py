from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pipeline import MultimodalIngestionPipeline


PDF_DIR = PROJECT_ROOT / "Data" / "Pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parsing-only audit over PDF chunks.")
    parser.add_argument("--pdf", help="Optional single PDF path to audit.")
    args = parser.parse_args()

    pipeline = MultimodalIngestionPipeline()
    counter: Counter[str] = Counter()
    missing_chapter_title = 0
    missing_visual_entity_id = 0
    verified_asset_paths = 0
    missing_asset_paths = 0
    mixed_table_figure = 0
    missing_section_title = 0
    missing_subsection_title = 0
    blocked_asset_paths = 0
    parser_counts: Counter[str] = Counter()
    top_visuals: list[dict[str, object]] = []

    pdf_paths = [Path(args.pdf)] if args.pdf else sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_paths:
        result = pipeline.ingest_sync(pdf_path)
        for chunk in result.chunks:
            metadata = dict(chunk.metadata)
            counter["total_pdf_chunks"] += 1
            counter[f"chunk_type::{metadata.get('chunk_type', 'unknown')}"] += 1
            counter[f"entity_type::{metadata.get('entity_type', 'unknown')}"] += 1
            parser_counts[str(metadata.get("parser") or result.metadata.get("parser") or "unknown")] += 1
            if not metadata.get("chapter_title") and metadata.get("document_type") == "pdf":
                missing_chapter_title += 1
            if not metadata.get("section_title") and metadata.get("document_type") == "pdf":
                missing_section_title += 1
            if metadata.get("chunk_type") == "section_text_chunk" and not metadata.get("subsection_title"):
                missing_subsection_title += 1
            if metadata.get("entity_type") in {"figure", "chart", "diagram", "image", "map", "table"} and not metadata.get("entity_id"):
                missing_visual_entity_id += 1
            if metadata.get("asset_paths"):
                verified_asset_paths += 1
            elif metadata.get("chunk_type") == "visual_asset_chunk":
                missing_asset_paths += 1
            if metadata.get("chunk_type") == "visual_asset_chunk" and metadata.get("asset_validation_status") == "blocked":
                blocked_asset_paths += 1
            if metadata.get("contains_table") and (
                metadata.get("figure_image_path")
                or metadata.get("chart_image_path")
                or metadata.get("diagram_image_path")
            ):
                mixed_table_figure += 1
            if metadata.get("chunk_type") == "visual_asset_chunk":
                top_visuals.append(
                    {
                        "entity_id": metadata.get("entity_id"),
                        "page_no": metadata.get("page_no"),
                        "title": metadata.get("visual_title"),
                        "chapter": metadata.get("chapter_title"),
                        "asset_path": (metadata.get("asset_paths") or [""])[0],
                    }
                )

    print("PDF Chunk Audit")
    print("===============")
    print(f"total PDF chunks: {counter['total_pdf_chunks']}")
    print(f"total chapter chunks: {counter['chunk_type::chapter_chunk']}")
    print(f"total section text chunks: {counter['chunk_type::section_text_chunk']}")
    print(f"total visual caption chunks: {counter['chunk_type::visual_caption_chunk']}")
    print(f"total visual asset chunks: {counter['chunk_type::visual_asset_chunk']}")
    print(f"total visual context chunks: {counter['chunk_type::visual_context_chunk']}")
    print(f"total outline chunks: {counter['chunk_type::document_outline_chunk']}")
    print(f"total figure chunks: {counter['entity_type::figure']}")
    print(f"total chart chunks: {counter['entity_type::chart']}")
    print(f"total diagram chunks: {counter['entity_type::diagram']}")
    print(f"total image chunks: {counter['entity_type::image']}")
    print(f"total map chunks: {counter['entity_type::map']}")
    print(f"total table chunks: {counter['entity_type::table']}")
    print(f"chunks with chapter_title missing: {missing_chapter_title}")
    print(f"chunks with section_title missing: {missing_section_title}")
    print(f"section text chunks with subsection_title missing: {missing_subsection_title}")
    print(f"chunks with entity_id missing where entity_type is visual/table: {missing_visual_entity_id}")
    print(f"visual chunks with verified asset paths: {verified_asset_paths}")
    print(f"visual chunks without asset paths: {missing_asset_paths}")
    print(f"visual chunks with blocked asset validation: {blocked_asset_paths}")
    print(f"chunks with mixed table/figure metadata: {mixed_table_figure}")
    print(f"parser counts: {dict(parser_counts)}")
    print("")
    print("Top 20 visual entities")
    for item in top_visuals[:20]:
        print(item)


if __name__ == "__main__":
    main()
