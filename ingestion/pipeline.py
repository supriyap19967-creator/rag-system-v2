from __future__ import annotations

import argparse
import asyncio
import logging
import re
from pathlib import Path

from ingestion.chunking import MarkdownChunker
from ingestion.config import IngestionSettings
from ingestion.detect_figures import FigureDetector
from ingestion.extract_images import ImageExtractor
from ingestion.merge_content import ContentMerger
from ingestion.parse_csv import CsvSemanticParser
from ingestion.parse_pdf import DoclingPdfParser
from ingestion.schemas import EnrichedDocument, ExtractedImage, IngestionResult
from ingestion.vision_caption import VisionCaptioner


logger = logging.getLogger(__name__)


def _safe_asset_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "document"


def _page_no_from_asset_name(path: Path) -> int | None:
    match = re.search(r"page[_-]?(?P<page>\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group("page")) if match else None


def _type_from_asset_name(path: Path) -> str:
    stem = path.stem.lower()
    if "table" in stem:
        return "table"
    if "map" in stem:
        return "map"
    if "diagram" in stem:
        return "diagram"
    if "chart" in stem or "graph" in stem:
        return "chart"
    return "figure"


class MultimodalIngestionPipeline:
    """End-to-end ingestion pipeline for multimodal conversational RAG."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or IngestionSettings()
        self.pdf_parser = DoclingPdfParser(self.settings)
        self.csv_parser = CsvSemanticParser(self.settings)
        self.image_extractor = ImageExtractor(FigureDetector(self.settings))
        self.captioner = VisionCaptioner(self.settings)
        self.merger = ContentMerger()
        self.chunker = MarkdownChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    @staticmethod
    def load_vision_models(settings: IngestionSettings | None = None) -> None:
        """Load PaddleOCR on CPU and Qwen AWQ on GPU, then stop."""
        VisionCaptioner(settings or IngestionSettings()).warmup()
        logger.info("Vision models loaded (PaddleOCR=CPU, Qwen=GPU). Stopping.")

    async def ingest(self, source_path: str | Path) -> IngestionResult:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        suffix = source_path.suffix.lower()
        logger.info("Starting multimodal ingestion for %s", source_path)

        if suffix == ".pdf":
            enriched = await self._ingest_pdf(source_path)
        elif suffix == ".csv":
            enriched = await asyncio.to_thread(self.csv_parser.parse, source_path)
        else:
            raise ValueError(f"Unsupported file type: {source_path.suffix}")

        chunks = self.chunker.chunk(enriched)
        logger.info("Finished ingestion for %s with %s chunks", source_path, len(chunks))
        return IngestionResult(
            enriched_document=enriched,
            chunks=chunks,
            metadata={
                "source": str(source_path),
                "chunk_count": len(chunks),
                **enriched.metadata,
            },
        )

    async def _ingest_pdf(self, pdf_path: Path) -> EnrichedDocument:
        logger.info("Parsing PDF with Docling: %s", pdf_path)
        document = await asyncio.to_thread(self.pdf_parser.parse, pdf_path)

        images: list[ExtractedImage] = []
        descriptions = []
        if self.settings.extract_figures:
            images = self._existing_assets_for_pdf(pdf_path)
            if images:
                logger.info("Reusing %s existing extracted visual asset(s) for %s", len(images), pdf_path)
            else:
                try:
                    images = await asyncio.to_thread(self.image_extractor.extract_from_pdf, pdf_path)
                    logger.info("Extracted %s visual regions from %s", len(images), pdf_path)
                except Exception as exc:
                    logger.warning("Figure extraction failed for %s: %s", pdf_path, exc)

            if images and self.settings.use_vision:
                descriptions = await self.captioner.describe_images(images)
                logger.info("Generated %s vision descriptions for %s", len(descriptions), pdf_path)
        else:
            logger.info("Figure extraction explicitly disabled for %s via INGESTION_EXTRACT_FIGURES", pdf_path)

        return self.merger.merge(document, images, descriptions)

    def _existing_assets_for_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        base_dir = self.settings.figure_output_dir
        candidates: list[Path] = []
        report_dir = base_dir / _safe_asset_stem(pdf_path.stem)
        if report_dir.exists():
            candidates.extend(sorted(path for path in report_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}))
        if not candidates and base_dir.exists():
            candidates.extend(
                sorted(
                    path for path in base_dir.glob("page*.*")
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and ".raw" not in path.name.lower()
                )
            )
        output: list[ExtractedImage] = []
        for path in candidates:
            resolved = path.resolve()
            page_no = _page_no_from_asset_name(path)
            entity_id = path.stem
            output.append(
                ExtractedImage(
                    image_path=resolved,
                    page=page_no,
                    type=_type_from_asset_name(path),
                    source_path=str(pdf_path),
                    element_id=entity_id,
                    metadata={
                        "source_label": path.stem.replace("_", " "),
                        "entity_id": entity_id,
                        "image_path": str(resolved),
                        "category": "existing_asset",
                    },
                )
            )
        return output

    def ingest_sync(self, source_path: str | Path) -> IngestionResult:
        return asyncio.run(self.ingest(source_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal ingestion for a PDF or CSV file.")
    parser.add_argument("source", nargs="?", help="Path to a PDF or CSV file.")
    parser.add_argument(
        "--load-models-only",
        action="store_true",
        help="Load PaddleOCR (CPU) and Qwen AWQ (GPU), print progress, then exit.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")

    if args.load_models_only:
        MultimodalIngestionPipeline.load_vision_models()
        print("Models loaded (PaddleOCR=CPU, Qwen=GPU). Stopping.")
        return

    if not args.source:
        parser.print_help()
        return

    result = MultimodalIngestionPipeline().ingest_sync(args.source)
    print(f"Ingestion completed: {len(result.chunks)} chunks")
    print(f"Metadata: {result.metadata}")
    for index, chunk in enumerate(result.chunks[:3], start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk.text[:1000])


if __name__ == "__main__":
    main()
