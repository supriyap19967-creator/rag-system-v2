from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.csv_chunking import parse_csv_file

CSV_DIR = PROJECT_ROOT / "Data" / "csv"
EXTRACTED_TABLE_DIR = PROJECT_ROOT / "assets" / "extracted_tables"


def main() -> None:
    paths = [
        *sorted(CSV_DIR.glob("*.csv")),
        *sorted(EXTRACTED_TABLE_DIR.glob("*.csv")),
    ]
    counter: Counter[str] = Counter()
    missing_country_code = 0
    missing_indicator_code = 0
    mixed_table_figure = 0
    example_chunks: dict[str, dict[str, object]] = {}

    for path in paths:
        parsed = parse_csv_file(path)
        counter[f"files::{parsed.csv_kind}"] += 1
        for block in parsed.blocks:
            metadata = dict(block.metadata)
            entity_type = str(metadata.get("entity_type") or "unknown")
            counter[f"chunks::{entity_type}"] += 1
            if metadata.get("csv_path"):
                counter["has_csv_path"] += 1
            if metadata.get("table_csv_path"):
                counter["has_table_csv_path"] += 1
            if metadata.get("contains_table"):
                counter["contains_table"] += 1
            if metadata.get("contains_figure"):
                counter["contains_figure"] += 1
            if not metadata.get("country_code") and entity_type in {"csv_timeseries", "csv_timeseries_range", "country_metadata"}:
                missing_country_code += 1
            if not metadata.get("indicator_code") and entity_type in {"csv_timeseries", "csv_timeseries_range", "indicator_metadata"}:
                missing_indicator_code += 1
            if metadata.get("table_csv_path") and (metadata.get("figure_image_path") or metadata.get("chart_image_path")):
                mixed_table_figure += 1
            example_chunks.setdefault(entity_type, {"text": block.text, "metadata": metadata})

    print("CSV Chunk Audit")
    print("================")
    print(f"total standalone CSV chunks: {counter['chunks::csv_timeseries'] + counter['chunks::csv_timeseries_range']}")
    print(f"total country metadata chunks: {counter['chunks::country_metadata']}")
    print(f"total indicator metadata chunks: {counter['chunks::indicator_metadata']}")
    print(f"total extracted table CSV chunks: {counter['chunks::table']}")
    print(f"number of chunks with csv_path: {counter['has_csv_path']}")
    print(f"number of chunks with table_csv_path: {counter['has_table_csv_path']}")
    print(f"number of chunks missing country_code where expected: {missing_country_code}")
    print(f"number of chunks missing indicator_code where expected: {missing_indicator_code}")
    print(f"number of chunks with mixed table/figure metadata: {mixed_table_figure}")
    print("")
    for entity_type, example in sorted(example_chunks.items()):
        print(f"[example] {entity_type}")
        print(example["text"])
        print(example["metadata"])
        print("")


if __name__ == "__main__":
    main()
