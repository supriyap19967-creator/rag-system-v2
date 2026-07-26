import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from ingestion.csv_chunking import parse_csv_file


logger = logging.getLogger(__name__)

CSV_PATH = Path("Data/csv/GDP1.csv")
DEFAULT_CSV_DIR = Path("Data/csv")
DEFAULT_EXTRACTED_TABLE_CSV_DIR = Path("assets/extracted_tables")
DEFAULT_PDF_DIR = Path("Data/Pdf")
START_PAGE = int(os.getenv("PDF_INGESTION_START_PAGE", os.getenv("PDF_VISUAL_START_PAGE", "4")))
END_PAGE = int(os.getenv("PDF_INGESTION_END_PAGE", os.getenv("PDF_VISUAL_END_PAGE", "5")))
PDF_CHUNK_SIZE = 1200
PDF_CHUNK_OVERLAP = 150
PDF_MIN_PARAGRAPH_LENGTH = 45
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
CITATION_HEAVY_PATTERN = re.compile(
    r"\b(?:doi|journal|press|vol\.?|no\.?|pp\.?|isbn|issn|retrieved|available at)\b",
    re.IGNORECASE,
)


def infer_metric_family(indicator: object, dataset_type: object = None) -> str:
    normalized = " ".join(
        part
        for part in (
            str(indicator or "").lower(),
            str(dataset_type or "").lower(),
        )
        if part
    )
    if "gdp" in normalized or "ny.gdp.mktp.cd" in normalized:
        return "gdp"
    if (
        "co2" in normalized
        or "carbon dioxide" in normalized
        or "emission" in normalized
        or "en.ghg.co2" in normalized
    ):
        return "co2"
    return ""


def _iter_csv_paths(csv_dir: Optional[Path], csv_path: Path) -> Iterable[Path]:
    if csv_dir is not None:
        yield from sorted(path for path in csv_dir.glob("*.csv") if path.is_file())
        return
    yield csv_path


def _load_world_bank_csv_documents(csv_path: Path) -> List[Document]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    parsed = parse_csv_file(csv_path)
    documents: List[Document] = []
    for record in parsed.blocks:
        metadata = dict(record.metadata)
        metadata.setdefault("source", str(csv_path))
        metadata.setdefault("source_files", csv_path.name)
        metadata.setdefault("source_type", metadata.get("document_type", parsed.metadata.get("source_type", "csv")))
        metadata.setdefault("dataset_type", metadata.get("indicator_code") or metadata.get("indicator_name") or parsed.csv_kind)
        metadata.setdefault("country_iso3", metadata.get("country_code", ""))
        metadata.setdefault("indicator", metadata.get("indicator_name", ""))
        metadata.setdefault(
            "metric_family",
            infer_metric_family(metadata.get("indicator_name"), metadata.get("indicator_code")),
        )
        documents.append(Document(page_content=record.text, metadata=metadata))
    return documents


def load_ingestion_documents(
    csv_dir: Optional[Path] = None,
    csv_path: Path = CSV_PATH,
    pdf_dir: Optional[Path] = None,
    include_csv_vectors: bool = False,
    include_pdf_visuals: bool = True,
    table_csv_dir: Optional[Path] = DEFAULT_EXTRACTED_TABLE_CSV_DIR,
) -> List[Document]:
    documents: List[Document] = []
    if include_csv_vectors:
        logger.warning(
            "CSV vector chunking is deprecated; use app.structured_query.PandasStructuredQueryEngine for numeric answers."
        )
        for path in _iter_csv_paths(csv_dir, csv_path):
            documents.extend(_load_world_bank_csv_documents(path))
        if table_csv_dir is not None and table_csv_dir.exists():
            for path in sorted(table_csv_dir.glob("*.csv")):
                documents.extend(_load_world_bank_csv_documents(path))
    if pdf_dir is not None:
        documents.extend(load_pdf_documents(pdf_dir))
        if include_pdf_visuals:
            try:
                from app.pdf_visual_extraction import extract_pdf_visual_documents

                documents.extend(extract_pdf_visual_documents(pdf_dir))
            except Exception as exc:
                logger.warning("PDF visual document extraction skipped: %s", exc)
    return documents


def _chunk_text(
    text: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP,
) -> List[str]:
    paragraphs = _extract_pdf_paragraphs(text)
    if not paragraphs:
        return []

    return _paragraphs_to_chunks(paragraphs, chunk_size=chunk_size, overlap=overlap)


def _join_hyphenated_match(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if right.lower() in {"and", "or", "to", "from", "with"}:
        return f"{left}- {right}"
    return f"{left}{right}"


def _clean_pdf_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", r"\1\2", cleaned)
    cleaned = re.sub(r"([A-Za-z]{2,})\s*-\s+([A-Za-z]{2,})", _join_hyphenated_match, cleaned)

    cleaned_lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        line = re.sub(r"\s*/\s*", "/", line)
        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_page_number_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if re.fullmatch(r"(?:page\s+)?\d{1,4}\s*(?:of\s*\d{1,4})?", stripped.lower()):
        return True
    if re.fullmatch(r"[ivxlcdm]{1,8}", stripped.lower()):
        return True
    return False


def _is_noisy_pdf_line(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if URL_PATTERN.search(stripped):
        return True
    if _is_page_number_fragment(stripped):
        return True
    if re.search(r"\b(?:contents|table of contents|references|bibliography|acknowledg(?:e)?ments|foreword)\b", lowered):
        return True
    if re.search(r"\.{4,}", stripped):
        return True
    if re.fullmatch(r"(?:chapter|section|figure|table)\s+\d+[a-z]?", lowered):
        return True
    if re.fullmatch(r"world development report\s+\d{4}\s+\d{1,4}", lowered):
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", stripped)
    digit_tokens = re.findall(r"\d+", stripped)
    if digit_tokens and len(digit_tokens) >= max(len(alpha_tokens), 3):
        return True
    return False


def _is_noisy_pdf_paragraph(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.lower()
    if not normalized or len(normalized) < PDF_MIN_PARAGRAPH_LENGTH:
        return True
    if URL_PATTERN.search(normalized):
        return True
    if _is_page_number_fragment(normalized):
        return True
    if CITATION_HEAVY_PATTERN.search(normalized) and len(re.findall(r"[.!?]", normalized)) <= 1:
        return True
    if re.search(r"\b(?:contents|references|bibliography|acknowledg(?:e)?ments|foreword)\b", lowered):
        return True
    if re.search(r"\b(?:figure|table)\s+\d+\b", lowered):
        return True
    words = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", normalized)
    if words:
        digit_ratio = sum(1 for token in words if re.search(r"\d", token)) / len(words)
        if digit_ratio > 0.35:
            return True
    return False


def _should_continue_paragraph(previous_line: str, current_line: str) -> bool:
    if not previous_line:
        return False
    if previous_line.endswith(("-", "/", ",")):
        return True
    if not re.search(r"[.!?:]$", previous_line):
        return True
    if current_line and current_line[:1].islower():
        return True
    return False


def _extract_pdf_paragraphs(text: str) -> List[str]:
    cleaned = _clean_pdf_text(text)
    if not cleaned:
        return []

    lines = cleaned.splitlines()
    paragraphs: List[str] = []
    current_lines: List[str] = []

    def flush() -> None:
        if not current_lines:
            return
        paragraph = re.sub(r"\s+", " ", " ".join(current_lines)).strip()
        current_lines.clear()
        if not _is_noisy_pdf_paragraph(paragraph):
            paragraphs.append(paragraph)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _is_noisy_pdf_line(stripped):
            flush()
            continue
        if current_lines and not _should_continue_paragraph(current_lines[-1], stripped):
            flush()
        current_lines.append(stripped)
    flush()
    return paragraphs


def _paragraphs_to_chunks(
    paragraphs: List[str],
    chunk_size: int,
    overlap: int,
) -> List[str]:
    chunks: List[str] = []
    current_parts: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_parts and current_length + paragraph_length + 1 > chunk_size:
            chunk = "\n\n".join(current_parts).strip()
            if chunk:
                chunks.append(chunk)

            overlap_parts: List[str] = []
            overlap_length = 0
            for part in reversed(current_parts):
                projected = overlap_length + len(part) + (2 if overlap_parts else 0)
                if projected > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_length = projected
            current_parts = overlap_parts[:]
            current_length = sum(len(part) for part in current_parts)
            if current_parts:
                current_length += 2 * (len(current_parts) - 1)

        current_parts.append(paragraph)
        current_length += paragraph_length + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        chunk = "\n\n".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def load_pdf_documents(pdf_dir: Path = DEFAULT_PDF_DIR) -> List[Document]:
    if not pdf_dir.exists():
        return []

    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        logger.warning("PDF loading skipped because unstructured[pdf] is not installed: %s", exc)
        return []

    documents: List[Document] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"--- Starting text partitioning for pages {START_PAGE}-{END_PAGE}: {pdf_path.name} ---", flush=True)
        temp_dir = None
        try:
            selected_pdf_path, temp_dir = _selected_page_pdf(pdf_path, START_PAGE, END_PAGE)
            elements = partition_pdf(
                filename=str(selected_pdf_path),
                strategy="hi_res",
                infer_table_structure=True,
            )
        except Exception as exc:
            logger.warning("Unstructured PDF loading skipped for %s: %s", pdf_path, exc)
            continue
        finally:
            if temp_dir is not None:
                try:
                    temp_dir.cleanup()
                except Exception:
                    pass

        page_text: dict[int, List[str]] = {}
        for element in elements:
            category = str(getattr(element, "category", "") or element.__class__.__name__)
            if category in {"Image", "FigureCaption", "Table"}:
                continue
            text = str(element or "").strip()
            if not text:
                continue
            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None)
            if page_number is None and isinstance(metadata, dict):
                page_number = metadata.get("page_number")
            try:
                page_index = int(page_number or 1) + START_PAGE - 1
            except (TypeError, ValueError):
                page_index = START_PAGE
            page_text.setdefault(page_index, []).append(text)

        for page_index in sorted(page_text):
            text = "\n\n".join(page_text[page_index])
            for chunk_index, chunk in enumerate(_chunk_text(text), start=1):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(pdf_path),
                            "source_files": pdf_path.name,
                            "source_type": "pdf",
                            "dataset_type": "pdf",
                            "content_type": "text",
                            "element_type": "text",
                            "metric_family": "",
                            "page": page_index,
                            "source_page": page_index,
                            "chunk_index": chunk_index,
                        },
                    )
                )
    return documents


def _selected_page_pdf(pdf_path: Path, start_page: int, end_page: int) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory()
    selected_pdf_path = Path(temp_dir.name) / f"{pdf_path.stem}-pages-{start_page}-{end_page}.pdf"
    try:
        import fitz

        source = fitz.open(str(pdf_path))
        selected = fitz.open()
        first_index = max(start_page - 1, 0)
        last_index = min(end_page - 1, len(source) - 1)
        if first_index <= last_index:
            selected.insert_pdf(source, from_page=first_index, to_page=last_index)
            selected.save(str(selected_pdf_path))
        selected.close()
        source.close()
    except Exception as exc:
        temp_dir.cleanup()
        raise RuntimeError(f"Could not create selected-page PDF for {pdf_path}: {exc}") from exc
    return selected_pdf_path, temp_dir


if __name__ == "__main__":
    loaded_documents = load_ingestion_documents(
        csv_dir=DEFAULT_CSV_DIR,
        pdf_dir=DEFAULT_PDF_DIR,
    )
    csv_count = sum(1 for doc in loaded_documents if doc.metadata.get("source_type") == "csv")
    pdf_count = sum(1 for doc in loaded_documents if doc.metadata.get("source_type") == "pdf")
    logger.info(
        "Loaded %s documents (%s CSV, %s PDF).",
        len(loaded_documents),
        csv_count,
        pdf_count,
    )
