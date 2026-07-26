from __future__ import annotations

from pathlib import Path

from ingestion.config import IngestionSettings
from ingestion.csv_chunking import parse_csv_file
from ingestion.llama_parse_csv import LlamaParseCsvAdapter
from ingestion.schemas import ContentBlock, EnrichedDocument


class CsvSemanticParser:
    """Parse CSVs into deterministic structured chunks for retrieval."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or IngestionSettings()
        self._llama_adapter = LlamaParseCsvAdapter(self.settings)

    def parse(self, csv_path: Path) -> EnrichedDocument:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        extra_metadata = {}
        backend = self.settings.csv_backend
        if backend in {"auto", "llamaparse", "llama", "llama_parse"}:
            try:
                extra_metadata = self._llama_adapter.parse_markdown(csv_path)
            except Exception as exc:
                if backend in {"llamaparse", "llama", "llama_parse"}:
                    raise
                extra_metadata = {"llamaparse_used": False, "llamaparse_error": str(exc)}
        parsed = parse_csv_file(csv_path)
        blocks = [
            ContentBlock(
                text=record.text,
                type="table" if record.metadata.get("entity_type") == "table" else "csv_row",
                page=record.metadata.get("page_no"),
                source_path=str(csv_path),
                metadata=dict(record.metadata),
            )
            for record in parsed.blocks
        ]
        return EnrichedDocument(
            source_path=str(csv_path),
            markdown="\n\n".join(block.text for block in blocks).strip() + ("\n" if blocks else ""),
            blocks=blocks,
            metadata={**dict(parsed.metadata), **extra_metadata},
        )
