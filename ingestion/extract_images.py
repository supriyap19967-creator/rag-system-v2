from __future__ import annotations

from pathlib import Path

from ingestion.detect_figures import FigureDetector
from ingestion.schemas import ExtractedImage


class ImageExtractor:
    """Extract image crops from PDF visual elements."""

    def __init__(self, detector: FigureDetector | None = None) -> None:
        self.detector = detector or FigureDetector()

    def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        return self.detector.detect(pdf_path)
