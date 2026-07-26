from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ingestion.pipeline import MultimodalIngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal ingestion for a PDF or CSV.")
    parser.add_argument("source", type=Path, help="Path to a PDF or CSV file.")
    parser.add_argument("--out", type=Path, default=Path("Data/processed"), help="Output directory.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = MultimodalIngestionPipeline()
    result = pipeline.ingest_sync(args.source)

    stem = args.source.stem
    markdown_path = args.out / f"{stem}.enriched.md"
    metadata_path = args.out / f"{stem}.metadata.json"
    chunks_path = args.out / f"{stem}.chunks.jsonl"

    markdown_path.write_text(result.enriched_document.markdown, encoding="utf-8")
    metadata_path.write_text(json.dumps(result.metadata, indent=2, default=str), encoding="utf-8")
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in result.chunks:
            handle.write(json.dumps({"text": chunk.text, "metadata": chunk.metadata}, default=str) + "\n")

    print(f"Wrote {markdown_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {chunks_path}")


if __name__ == "__main__":
    main()
