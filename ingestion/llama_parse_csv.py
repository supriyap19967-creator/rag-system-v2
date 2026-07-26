from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ingestion.config import IngestionSettings


logger = logging.getLogger(__name__)


class LlamaParseCsvAdapter:
    """Optional LlamaParse adapter that enriches CSV parsing metadata without changing chunk shape."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or IngestionSettings()

    def parse_markdown(self, csv_path: Path) -> dict[str, Any]:
        api_key = self.settings.llama_parse_api_key.strip()
        if not api_key:
            raise RuntimeError("LlamaParse CSV parsing requested, but no LLAMA_CLOUD_API_KEY/LLAMA_PARSE_API_KEY is set.")
        try:
            from llama_parse import LlamaParse
        except ImportError as exc:
            raise RuntimeError("Install llama-parse to enable LlamaParse CSV parsing.") from exc

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction=(
                "Parse this CSV into clear markdown while preserving headers, row semantics, and data meaning. "
                "Do not hallucinate missing values."
            ),
        )
        try:
            documents = parser.load_data(str(csv_path))
        except TypeError:
            documents = parser.load_data(file_path=str(csv_path))

        texts: list[str] = []
        for doc in documents or []:
            text = str(getattr(doc, "text", "") or "").strip()
            if text:
                texts.append(text)

        markdown = "\n\n".join(texts).strip()
        logger.info("Parsed CSV via LlamaParse: %s", csv_path)
        return {
            "llamaparse_used": True,
            "llamaparse_markdown": markdown,
            "llamaparse_document_count": len(documents or []),
        }
