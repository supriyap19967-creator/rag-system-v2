from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

from ingestion.config import IngestionSettings
from ingestion.schemas import ExtractedImage
from ingestion.visual_paths import absolute_asset_path, canonical_flat_image_path, entity_token_from_label, safe_token


logger = logging.getLogger(__name__)

VISUAL_TYPES = {"Image", "Figure", "FigureCaption", "Chart", "Diagram", "Map", "Table"}
UNSTRUCTURED_IMAGE_BLOCK_TYPES = ["Image", "Table", "Figure"]


def _metadata_value(metadata: object, key: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _element_type(element: object) -> str:
    category = str(getattr(element, "category", "") or element.__class__.__name__)
    text = str(element or "").lower()
    if "table" in category.lower():
        return "table"
    if "map" in text or "map" in category.lower():
        return "map"
    if "chart" in text or "graph" in text:
        return "chart"
    if "diagram" in text or "flow" in text:
        return "diagram"
    if "image" in category.lower():
        return "image"
    return "figure"


def _element_text(element: object) -> str:
    return str(element or "").strip()


def _looks_like_figure_label(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:figure|fig\.?|chart|diagram|table|map|box|spotlight)\s+[A-Za-z0-9.:-]+",
            text,
            flags=re.IGNORECASE,
        )
    )


def _nearby_figure_label(elements: list[object], index: int) -> str:
    """Find the closest caption/label text adjacent to an extracted visual."""

    current_text = _element_text(elements[index])
    if current_text:
        return current_text

    for offset in (-1, 1, -2, 2):
        neighbor_index = index + offset
        if neighbor_index < 0 or neighbor_index >= len(elements):
            continue
        neighbor = elements[neighbor_index]
        category = str(getattr(neighbor, "category", "") or neighbor.__class__.__name__)
        text = _element_text(neighbor)
        if not text:
            continue
        if category == "FigureCaption" or _looks_like_figure_label(text):
            return text

    return ""


def _persist_image_bytes(image_bytes: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(image_bytes)
    return destination.resolve()


def _finalize_extracted_image(
    settings: IngestionSettings,
    pdf_path: Path,
    source_path: Path,
    *,
    page_number: int | None,
    visual_type: str,
    entity_label: str,
    element_id: str,
    category: str,
    coordinates: dict[str, Any],
    extraction_method: str,
) -> ExtractedImage | None:
    if not source_path.exists():
        logger.warning("Skipping visual element %s because image file does not exist: %s", element_id, source_path)
        return None

    canonical_path = canonical_flat_image_path(
        settings.figure_output_dir.resolve(),
        source_path,
        page_number=page_number,
        visual_type=visual_type,
        entity_label=entity_label or element_id,
    )
    entity_id = entity_token_from_label(entity_label or element_id, fallback=safe_token(element_id))
    return ExtractedImage(
        image_path=canonical_path,
        page=int(page_number) if page_number else None,
        type=visual_type,  # type: ignore[arg-type]
        source_path=str(pdf_path),
        element_id=entity_id,
        coordinates=coordinates,
        metadata={
            "category": category,
            "text": entity_label,
            "source_label": entity_label,
            "entity_id": entity_id,
            "image_path": absolute_asset_path(canonical_path),
            "extraction_method": extraction_method,
        },
    )


class FigureDetector:
    """Unstructured-based figure, chart, table, and image-region detector."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or IngestionSettings()

    def detect(self, pdf_path: Path) -> list[ExtractedImage]:
        pdf_path = Path(pdf_path)
        scratch_dir = self.settings.figure_output_dir.resolve() / safe_token(pdf_path.stem)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        images = self._detect_with_unstructured(pdf_path, scratch_dir)
        if not images:
            logger.info("Unstructured found no visuals in %s; falling back to PyMuPDF embedded-image extraction", pdf_path)
            images = self._detect_with_pymupdf(pdf_path, scratch_dir)

        logger.info("Detected %s visual regions in %s", len(images), pdf_path)
        return images

    def _detect_with_unstructured(self, pdf_path: Path, scratch_dir: Path) -> list[ExtractedImage]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise RuntimeError("unstructured[pdf] is required for figure detection.") from exc

        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=self.settings.pdf_strategy,
                infer_table_structure=True,
                extract_image_block_types=UNSTRUCTURED_IMAGE_BLOCK_TYPES,
                extract_image_block_output_dir=str(scratch_dir),
            )
        except Exception as exc:
            logger.warning("Unstructured figure detection failed for %s: %s", pdf_path, exc)
            return []

        images: list[ExtractedImage] = []
        for zero_index, element in enumerate(elements):
            index = zero_index + 1
            category = str(getattr(element, "category", "") or element.__class__.__name__)
            if category not in VISUAL_TYPES:
                continue

            metadata = getattr(element, "metadata", None)
            image_path = _metadata_value(metadata, "image_path")
            image_base64 = _metadata_value(metadata, "image_base64")
            page_number = _metadata_value(metadata, "page_number")
            coordinates = _metadata_value(metadata, "coordinates") or {}
            entity_label = _nearby_figure_label(elements, zero_index) or _element_text(element)
            visual_type = _element_type(element)
            element_id = str(_metadata_value(metadata, "element_id") or f"visual-{index}")

            if not image_path and image_base64:
                raw_path = scratch_dir / f"page{page_number or 'unknown'}_{index}.png"
                image_path = _persist_image_bytes(base64.b64decode(image_base64), raw_path)

            if not image_path:
                continue

            extracted = _finalize_extracted_image(
                self.settings,
                pdf_path,
                Path(image_path),
                page_number=int(page_number) if page_number else None,
                visual_type=visual_type,
                entity_label=entity_label,
                element_id=element_id,
                category=category,
                coordinates=coordinates.to_dict() if hasattr(coordinates, "to_dict") else dict(coordinates or {}),
                extraction_method="unstructured",
            )
            if extracted is not None:
                images.append(extracted)
        return images

    def _detect_with_pymupdf(self, pdf_path: Path, scratch_dir: Path) -> list[ExtractedImage]:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is unavailable; cannot run embedded-image fallback for %s", pdf_path)
            return []

        images: list[ExtractedImage] = []
        try:
            with fitz.open(str(pdf_path)) as document:
                for page_index, page in enumerate(document):
                    page_number = page_index + 1
                    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                        xref = image_info[0]
                        extracted = document.extract_image(xref)
                        image_bytes = extracted.get("image") if extracted else None
                        if not image_bytes:
                            continue
                        extension = str(extracted.get("ext") or "png").lower()
                        if extension == "jpg":
                            extension = "jpeg"
                        suffix = f".{extension}" if extension else ".png"
                        raw_path = scratch_dir / f"page_{page_number}_embedded_{image_index}{suffix}"
                        raw_path = _persist_image_bytes(image_bytes, raw_path)
                        entity_label = f"Figure page {page_number} image {image_index}"
                        finalized = _finalize_extracted_image(
                            self.settings,
                            pdf_path,
                            raw_path,
                            page_number=page_number,
                            visual_type="figure",
                            entity_label=entity_label,
                            element_id=f"page_{page_number}_image_{image_index}",
                            category="Image",
                            coordinates={},
                            extraction_method="pymupdf_embedded",
                        )
                        if finalized is not None:
                            images.append(finalized)
        except Exception as exc:
            logger.warning("PyMuPDF embedded-image extraction failed for %s: %s", pdf_path, exc)
        return images
