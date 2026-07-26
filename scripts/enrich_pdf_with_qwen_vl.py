from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ingestion.config import IngestionSettings
from ingestion.qwen_vision_caption import NARRATIVE_PROMPT, QwenVisionCaptioner
from ingestion.schemas import ExtractedImage


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"


@dataclass(frozen=True, slots=True)
class ExtractedFigure:
    figure_id: str
    image_path: Path
    placeholder: str


@dataclass(frozen=True, slots=True)
class DoclingParseResult:
    markdown: str
    figures: list[ExtractedFigure]


class DoclingPdfParser:
    def __init__(self, output_dir: Path, image_scale: float = 2.0) -> None:
        self.output_dir = output_dir
        self.image_scale = image_scale

    def parse_pdf(self, pdf_path: Path) -> DoclingParseResult:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import PictureItem

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.image_scale
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        document = result.document

        try:
            markdown = document.export_to_markdown()
        except Exception as exc:
            raise RuntimeError(f"Docling Markdown export failed for {pdf_path}: {exc}") from exc

        figures: list[ExtractedFigure] = []
        for figure_index, item in enumerate(self._iter_picture_items(document, PictureItem), start=1):
            figure_id = f"figure_{figure_index:04d}"
            image_path = self.output_dir / f"{pdf_path.stem}_{figure_id}.png"
            self._save_picture_item(item, document, image_path)
            placeholder = f"<!-- {figure_id.upper()}_ANCHOR: {image_path.as_posix()} -->"
            figures.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    image_path=image_path,
                    placeholder=placeholder,
                )
            )

        stitched_markdown = self._inject_figure_placeholders(markdown, figures)
        return DoclingParseResult(markdown=stitched_markdown, figures=figures)

    @staticmethod
    def _iter_picture_items(document: object, picture_type: type) -> Iterable[object]:
        try:
            for item, _level in document.iterate_items():
                if isinstance(item, picture_type):
                    yield item
        except Exception as exc:
            logger.warning("Could not iterate Docling picture items: %s", exc)

    @staticmethod
    def _save_picture_item(item: object, document: object, image_path: Path) -> None:
        try:
            image = item.get_image(document)
            if image is None:
                raise ValueError("Docling returned an empty image object.")
            image.save(image_path, "PNG")
        except Exception as exc:
            raise RuntimeError(f"Could not save extracted figure to {image_path}: {exc}") from exc

    @staticmethod
    def _inject_figure_placeholders(markdown: str, figures: list[ExtractedFigure]) -> str:
        if not figures:
            return markdown

        image_pattern = re.compile(r"!\[[^\]]*]\([^)]+\)")
        matches = list(image_pattern.finditer(markdown))
        if matches:
            output_parts: list[str] = []
            last_index = 0
            for match, figure in zip(matches, figures):
                output_parts.append(markdown[last_index : match.start()])
                output_parts.append(figure.placeholder)
                last_index = match.end()
            output_parts.append(markdown[last_index:])
            if len(figures) > len(matches):
                output_parts.append("\n\n")
                output_parts.extend(figure.placeholder + "\n\n" for figure in figures[len(matches) :])
            return "".join(output_parts)

        figure_block = "\n\n".join(figure.placeholder for figure in figures)
        return f"{markdown.rstrip()}\n\n{figure_block}\n"


def enrich_markdown_with_visual_descriptions(
    markdown: str,
    figures: list[ExtractedFigure],
    captions: dict[str, str],
) -> str:
    enriched = markdown
    for figure in figures:
        caption = captions.get(figure.figure_id, "").strip()
        if not caption:
            caption = "No visual description was generated for this figure."
        wrapped_caption = (
            "--- START CHART VISUAL DESCRIPTION ---\n"
            f"{caption}\n"
            "--- END CHART VISUAL DESCRIPTION ---"
        )
        enriched = enriched.replace(figure.placeholder, wrapped_caption)
    return enriched


def parse_and_enrich_pdf(
    pdf_path: Path,
    charts_dir: Path = Path("./extracted_charts"),
    model_name: str = DEFAULT_MODEL_NAME,
    prompt: str = NARRATIVE_PROMPT,
    max_new_tokens: int = 512,
) -> str:
    parser = DoclingPdfParser(output_dir=charts_dir)
    parsed = parser.parse_pdf(pdf_path)

    if not parsed.figures:
        logger.warning("No figures were extracted from %s", pdf_path)
        return parsed.markdown

    settings = IngestionSettings()
    captioner = QwenVisionCaptioner(
        settings,
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    captioner.warmup()

    captions: dict[str, str] = {}
    for figure in parsed.figures:
        if not figure.image_path.exists():
            logger.warning("Missing extracted figure image: %s", figure.image_path)
            continue
        logger.info("Generating Qwen2.5-VL caption for %s", figure.image_path)
        try:
            image = ExtractedImage(
                image_path=figure.image_path.resolve(),
                page=None,
                type="figure",
                source_path=str(pdf_path),
                element_id=figure.figure_id,
            )
            result = captioner.describe_image(image)
            captions[figure.figure_id] = result.description if result else ""
        except Exception as exc:
            logger.exception("Qwen2.5-VL captioning failed for %s: %s", figure.image_path, exc)
            captions[figure.figure_id] = ""

    return enrich_markdown_with_visual_descriptions(parsed.markdown, parsed.figures, captions)


def write_output(markdown: str, output_path: Path | None) -> None:
    if output_path is None:
        print(markdown)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote enriched Markdown to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PDF with Docling, caption extracted charts with PaddleOCR + Qwen2.5-VL AWQ, and emit enriched Markdown."
    )
    parser.add_argument("pdf_path", type=Path, help="Local PDF file to parse.")
    parser.add_argument("--charts-dir", type=Path, default=Path("./extracted_charts"), help="Directory for extracted figure images.")
    parser.add_argument("--output", type=Path, default=None, help="Optional enriched Markdown output path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Qwen2.5-VL AWQ model name or local path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens generated per visual caption.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    enriched_markdown = parse_and_enrich_pdf(
        pdf_path=args.pdf_path,
        charts_dir=args.charts_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    write_output(enriched_markdown, args.output)


if __name__ == "__main__":
    main()
