from __future__ import annotations

from pathlib import Path

from app.multimodal_assets import (
    build_asset_registry,
    enrich_chunk_metadata,
    resolve_best_asset,
    validate_asset_path,
)
from vectordb.metadata_schema import normalize_payload


PROJECT_ROOT = Path(__file__).resolve().parent


def test_table_asset_metadata_is_enriched_and_preserved() -> None:
    table_dir = PROJECT_ROOT / "assets" / "extracted_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "page_999_Table_99.1.csv"
    csv_path.write_text("Metric,Value\nCoverage,100\n", encoding="utf-8")

    registry = build_asset_registry(PROJECT_ROOT)
    metadata = enrich_chunk_metadata(
        {"source_file": "synthetic.pdf", "document_type": "pdf", "page_no": 999},
        "Table 99.1 Synthetic metrics\n| Metric | Value |\n| --- | --- |\n| Coverage | 100 |",
        registry,
    )

    assert metadata["contains_table"] is True
    assert metadata["contains_csv"] is True
    assert metadata["table_csv_path"] == str(csv_path.resolve())
    assert str(csv_path.resolve()) in metadata["table_csv_paths"]

    payload = normalize_payload("Table 99.1 Synthetic metrics", metadata).to_qdrant_payload()
    assert payload["contains_table"] is True
    assert payload["contains_csv"] is True
    assert payload["table_csv_path"] == str(csv_path.resolve())
    assert payload["metadata"]["table_csv_path"] == str(csv_path.resolve())


def test_figure_asset_resolver_uses_verified_image_path() -> None:
    image_dir = PROJECT_ROOT / "assets" / "extracted_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "page_998_Figure_98.1.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    chunk = {
        "id": "chunk-figure",
        "content": "Figure 98.1 shows the synthetic visual.",
        "metadata": {
            "chunk_id": "chunk-figure",
            "entity_id": "Figure_98.1",
            "entity_ids": ["Figure_98.1"],
            "contains_figure": True,
            "contains_image": True,
            "image_path": str(image_path.resolve()),
        },
    }

    resolution = resolve_best_asset("show Figure 98.1", [chunk])

    assert resolution.ok is True
    assert resolution.renderer == "image"
    assert resolution.path == str(image_path.resolve())


def test_asset_validation_blocks_unsafe_path() -> None:
    result = validate_asset_path(PROJECT_ROOT / ".." / "outside.csv", "table")

    assert result.ok is False
    assert "outside the approved asset directories" in result.reason


def test_asset_resolver_reports_missing_metadata_path() -> None:
    chunk = {
        "id": "chunk-table",
        "content": "Table 77.7 exists in text but no asset path was stored.",
        "metadata": {
            "chunk_id": "chunk-table",
            "entity_id": "Table_77.7",
            "contains_table": True,
        },
    }

    resolution = resolve_best_asset("show Table 77.7", [chunk], registry=[])

    assert resolution.ok is False
    assert resolution.reason == "metadata was missing an asset path"

