from __future__ import annotations

import logging
import os
from pathlib import Path

from ingestion.config import IngestionSettings
from ingestion.schemas import EnrichedDocument
from ingestion.pdf_chunking import build_pdf_document


logger = logging.getLogger(__name__)


class DoclingPdfParser:
    """Docling-backed PDF parser that preserves structure as markdown."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or IngestionSettings()

    def parse(self, pdf_path: Path) -> EnrichedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            import fitz
            with fitz.open(str(pdf_path)) as doc:
                actual_page_count = doc.page_count
            result = self._convert_with_docling(pdf_path)
            document = result.document
            if len(document.pages) < actual_page_count:
                raise RuntimeError(f"Docling conversion was incomplete: {len(document.pages)}/{actual_page_count} pages processed.")
            markdown = document.export_to_markdown()
            return build_pdf_document(pdf_path, markdown, parser_name="docling")
        except ImportError:
            logger.warning("Docling is unavailable; falling back to PyMuPDF text extraction for %s", pdf_path)
        except Exception as exc:
            logger.warning("Docling parsing failed for %s; falling back to PyMuPDF/pypdf: %s", pdf_path, exc)

        markdown = self._fallback_markdown(pdf_path)
        return build_pdf_document(pdf_path, markdown, parser_name="fallback_text")

    def _configure_workspace_cache(self) -> Path:
        hf_home = self.settings.workspace_hf_home.resolve()
        hf_cache = self.settings.workspace_hf_cache.resolve()
        docling_artifacts = self.settings.docling_artifacts_dir.resolve()
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
        hf_cache.mkdir(parents=True, exist_ok=True)
        hf_home.mkdir(parents=True, exist_ok=True)
        docling_artifacts.mkdir(parents=True, exist_ok=True)
        return docling_artifacts

    def _convert_with_docling(self, pdf_path: Path):
        docling_artifacts = self._configure_workspace_cache()

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.utils.model_downloader import download_models

        download_models(
            output_dir=docling_artifacts,
            progress=False,
            with_layout=True,
            with_tableformer=False,
            with_tableformer_v2=False,
            with_code_formula=False,
            with_picture_classifier=False,
            with_smolvlm=False,
            with_granitedocling=False,
            with_granitedocling_mlx=False,
            with_granitedocling_2stage=False,
            with_smoldocling=False,
            with_smoldocling_mlx=False,
            with_granite_vision=False,
            with_granite_chart_extraction=False,
            with_granite_chart_extraction_v4=False,
            with_rapidocr=False,
            with_easyocr=False,
        )
        pipeline_options = PdfPipelineOptions(
            document_timeout=60.0,
            artifacts_path=docling_artifacts,
            accelerator_options=AcceleratorOptions(device="cpu", num_threads=2),
            do_picture_description=False,
            do_picture_classification=False,
            do_chart_extraction=False,
            do_ocr=False,
            force_backend_text=True,
            do_table_structure=False,
            generate_picture_images=False,
            generate_page_images=False,
            generate_table_images=False,
            layout_batch_size=1,
            ocr_batch_size=1,
            table_batch_size=1,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return converter.convert(str(pdf_path))

    def _fallback_markdown(self, pdf_path: Path) -> str:
        markdown = ""
        try:
            import fitz

            with fitz.open(str(pdf_path)) as document:
                markdown = "\n\n".join(page.get_text("text") for page in document)
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed for %s; falling back to pypdf: %s", pdf_path, exc)

        if markdown.strip():
            return markdown

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(f"Could not parse PDF text from {pdf_path}: {exc}") from exc
