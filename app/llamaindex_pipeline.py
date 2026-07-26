import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embeddings import BGE_MODEL_NAME, get_bge_embeddings
from app.ingestion import DEFAULT_CSV_DIR, DEFAULT_PDF_DIR, infer_metric_family
from app.llamaindex_embedding import BgeLlamaIndexEmbedding
from app.pdf_visual_extraction import extract_pdf_visual_documents
from app.utils import log_event


load_dotenv()

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_DIR = Path(os.getenv("LLAMAINDEX_STORAGE_DIR", "Data/llamaindex_storage"))
DEFAULT_DOCUMENT_CACHE = Path(os.getenv("LLAMAINDEX_DOCUMENT_CACHE_PATH", "Data/llamaindex_documents.json"))
DEFAULT_VISUAL_BOOTSTRAP_CACHE = Path(os.getenv("LLAMAINDEX_VISUAL_BOOTSTRAP_CACHE", "Data/bm25_documents.json"))
DEFAULT_MANIFEST_PATH = DEFAULT_STORAGE_DIR / "manifest.json"
PDF_START_PAGE = int(os.getenv("LLAMAINDEX_PDF_START_PAGE", os.getenv("PDF_INGESTION_START_PAGE", "60")))
PDF_END_PAGE = int(os.getenv("LLAMAINDEX_PDF_END_PAGE", os.getenv("PDF_INGESTION_END_PAGE", "400")))
PDF_CHUNK_SIZE = int(os.getenv("LLAMAINDEX_PDF_CHUNK_SIZE", "1200"))
PDF_CHUNK_OVERLAP = int(os.getenv("LLAMAINDEX_PDF_CHUNK_OVERLAP", "150"))
BOOTSTRAP_VISUALS_FROM_CACHE = os.getenv("LLAMAINDEX_BOOTSTRAP_VISUALS_FROM_CACHE", "true").strip().lower() in {"1", "true", "yes"}
INDEX_CSV_ROWS = os.getenv("LLAMAINDEX_INDEX_CSV_ROWS", "false").strip().lower() in {"1", "true", "yes"}
PDF_NAMESPACE = os.getenv("LLAMAINDEX_PDF_NAMESPACE", "pdf_text").strip() or "pdf_text"
VISUAL_NAMESPACE = os.getenv("LLAMAINDEX_VISUAL_NAMESPACE", "visual").strip() or "visual"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
COUNTRY_CODE_OVERRIDES = {
    "india": "IND",
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "us": "USA",
    "u s": "USA",
}


@dataclass(frozen=True)
class LlamaIndexPipelineSettings:
    csv_dir: Path = DEFAULT_CSV_DIR
    pdf_dir: Path = DEFAULT_PDF_DIR
    storage_dir: Path = DEFAULT_STORAGE_DIR
    document_cache_path: Path = DEFAULT_DOCUMENT_CACHE
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    visual_bootstrap_cache_path: Path = DEFAULT_VISUAL_BOOTSTRAP_CACHE
    include_csv: bool = True
    include_pdf_text: bool = True
    include_visuals: bool = True
    bootstrap_visuals_from_cache: bool = BOOTSTRAP_VISUALS_FROM_CACHE
    index_csv_rows: bool = INDEX_CSV_ROWS
    rebuild_storage: bool = True
    pdf_namespace: str = PDF_NAMESPACE
    visual_namespace: str = VISUAL_NAMESPACE
    pinecone_index_name: str = PINECONE_INDEX_NAME


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _metadata_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _current_embedding_dimension() -> int:
    return len(get_bge_embeddings().embed_query("dimension check"))


def _pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY in environment.")
    return Pinecone(api_key=PINECONE_API_KEY)


def _pinecone_index(index_name: str):
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME in environment.")
    client = _pinecone_client()
    index_names = set(client.list_indexes().names())
    if index_name not in index_names:
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")
    description = client.describe_index(index_name)
    expected_dimension = _current_embedding_dimension()
    actual_dimension = int(description.dimension)
    if actual_dimension != expected_dimension:
        raise RuntimeError(
            f"Pinecone index '{index_name}' has dimension {actual_dimension}, "
            f"but the current embedding model '{BGE_MODEL_NAME}' produces {expected_dimension}-dim vectors."
        )
    return client.Index(index_name)


def normalize_country_code(country_name: object, country_code: object = "") -> str:
    explicit_code = str(country_code or "").strip().upper()
    if explicit_code:
        return explicit_code
    normalized_name = re.sub(r"[^a-z0-9]+", " ", str(country_name or "").lower()).strip()
    if normalized_name in COUNTRY_CODE_OVERRIDES:
        return COUNTRY_CODE_OVERRIDES[normalized_name]
    try:
        import pycountry

        country = pycountry.countries.lookup(str(country_name or ""))
        return str(country.alpha_3).upper()
    except Exception:
        return ""


def _normalized_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    allowed_keys = {
        "source",
        "source_files",
        "source_type",
        "content_type",
        "element_type",
        "visual_type",
        "figure_id",
        "caption",
        "source_pdf",
        "page",
        "source_page",
        "image_path",
        "image_local_path",
        "dataset_type",
        "country_name",
        "country_iso3",
        "country_code",
        "indicator",
        "indicator_code",
        "metric_family",
        "year",
        "value",
        "row_index",
        "chunk_index",
        "section",
        "section_header",
        "topic",
        "crop_quality",
        "crop_quality_score",
        "crop_rejected_reason",
        "retrieval_group",
    }
    normalized: Dict[str, object] = {}
    for key, value in metadata.items():
        key = str(key)
        if key not in allowed_keys:
            continue
        normalized_value = _metadata_value(value)
        if isinstance(normalized_value, str) and len(normalized_value) > 500:
            normalized_value = normalized_value[:500]
        normalized[key] = normalized_value
    return normalized


def _iter_csv_paths(csv_dir: Path) -> Iterable[Path]:
    if not csv_dir.exists():
        return []
    return sorted(path for path in csv_dir.glob("*.csv") if path.is_file())


def _open_world_bank_csv(path: Path):
    handle = path.open("r", encoding="utf-8-sig", newline="")
    for _ in range(4):
        position = handle.tell()
        line = handle.readline()
        if not line:
            break
        if "Country Name" in line and "Country Code" in line:
            handle.seek(position)
            return handle
    return handle


def _load_csv_nodes(csv_dir: Path) -> List[LangchainDocument]:
    documents: List[LangchainDocument] = []
    for csv_path in _iter_csv_paths(csv_dir):
        try:
            handle = _open_world_bank_csv(csv_path)
            with handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                for row_index, row in enumerate(reader, start=1):
                    clean_row = {
                        str(key or "").strip(): str(value or "").strip()
                        for key, value in row.items()
                        if key
                    }
                    country_name = clean_row.get("Country Name", "")
                    country_iso3 = normalize_country_code(country_name, clean_row.get("Country Code", ""))
                    indicator = clean_row.get("Indicator Name", "")
                    indicator_code = clean_row.get("Indicator Code", "")
                    if not country_name and not indicator:
                        continue
                    for year, value in clean_row.items():
                        if not year.isdigit() or not value:
                            continue
                        text = f"In {year}, {indicator} for {country_name} ({country_iso3}) was {value}."
                        documents.append(
                            LangchainDocument(
                                page_content=text,
                                metadata={
                                    "source": str(csv_path),
                                    "source_files": csv_path.name,
                                    "source_type": "csv",
                                    "content_type": "table_data",
                                    "dataset_type": indicator_code or indicator,
                                    "country_name": country_name,
                                    "country_iso3": country_iso3,
                                    "country_code": country_iso3,
                                    "indicator": indicator,
                                    "indicator_code": indicator_code,
                                    "metric_family": infer_metric_family(indicator, indicator_code),
                                    "year": year,
                                    "value": value,
                                    "row_index": row_index,
                                    "retrieval_group": "csv_structured",
                                },
                            )
                        )
        except Exception as exc:
            logger.warning("Skipping CSV %s: %s", csv_path, exc)
    return documents


def _clean_pdf_text(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", r"\1\2", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _paragraphs(text: str) -> List[str]:
    paragraphs: List[str] = []
    for paragraph in re.split(r"\n\s*\n", _clean_pdf_text(text)):
        paragraph = _clean_text(paragraph)
        if len(paragraph) < 45:
            continue
        lowered = paragraph.lower()
        if re.search(r"\b(?:references|bibliography|contents|isbn|issn|doi)\b", lowered):
            continue
        if re.fullmatch(r"\d{1,4}", paragraph):
            continue
        paragraphs.append(paragraph)
    return paragraphs


def _chunk_paragraphs(paragraphs: Sequence[str]) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in paragraphs:
        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > PDF_CHUNK_SIZE:
            chunks.append("\n\n".join(current).strip())
            overlap: List[str] = []
            overlap_len = 0
            for item in reversed(current):
                if overlap_len + len(item) > PDF_CHUNK_OVERLAP:
                    break
                overlap.insert(0, item)
                overlap_len += len(item)
            current = overlap
            current_len = sum(len(item) for item in current)
        current.append(paragraph)
        current_len += len(paragraph) + (2 if len(current) > 1 else 0)
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def _load_pdf_text_nodes(pdf_dir: Path) -> List[LangchainDocument]:
    documents: List[LangchainDocument] = []
    if not pdf_dir.exists():
        return documents
    try:
        import fitz
    except ImportError as exc:
        logger.warning("PDF text ingestion skipped because PyMuPDF is unavailable: %s", exc)
        return documents

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.warning("Skipping PDF %s: %s", pdf_path, exc)
            continue
        try:
            first_page = max(PDF_START_PAGE, 1)
            last_page = min(PDF_END_PAGE, len(pdf))
            for page_number in range(first_page, last_page + 1):
                page = pdf[page_number - 1]
                chunks = _chunk_paragraphs(_paragraphs(page.get_text("text")))
                for chunk_index, chunk in enumerate(chunks, start=1):
                    documents.append(
                        LangchainDocument(
                            page_content=chunk,
                            metadata={
                                "source": str(pdf_path),
                                "source_files": pdf_path.name,
                                "source_type": "pdf",
                                "content_type": "text",
                                "dataset_type": "pdf",
                                "page": page_number,
                                "source_page": page_number,
                                "chunk_index": chunk_index,
                                "retrieval_group": "pdf_text",
                            },
                        )
                    )
        finally:
            pdf.close()
    return documents


def _load_visual_nodes(pdf_dir: Path) -> List[LangchainDocument]:
    if not pdf_dir.exists():
        return []
    try:
        documents = list(extract_pdf_visual_documents(pdf_dir=pdf_dir))
    except Exception as exc:
        logger.warning("Visual ingestion skipped: %s", exc)
        return []

    for document in documents:
        document.metadata["retrieval_group"] = "visual"
    return documents


def _load_visual_nodes_from_cache(cache_path: Path) -> List[LangchainDocument]:
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read visual bootstrap cache %s: %s", cache_path, exc)
        return []

    documents: List[LangchainDocument] = []
    for item in payload.get("documents", []):
        metadata = dict(item.get("metadata") or {})
        if str(metadata.get("content_type") or "").lower() != "visual":
            continue
        text = str(item.get("page_content") or item.get("text") or metadata.get("original_text") or "").strip()
        if not text:
            continue
        metadata["retrieval_group"] = "visual"
        documents.append(LangchainDocument(page_content=text, metadata=metadata))
    return documents


def collect_source_documents(settings: LlamaIndexPipelineSettings) -> List[LangchainDocument]:
    documents: List[LangchainDocument] = []
    if settings.include_csv:
        documents.extend(_load_csv_nodes(settings.csv_dir))
    if settings.include_pdf_text:
        documents.extend(_load_pdf_text_nodes(settings.pdf_dir))
    if settings.include_visuals:
        visual_documents = (
            _load_visual_nodes_from_cache(settings.visual_bootstrap_cache_path)
            if settings.bootstrap_visuals_from_cache
            else []
        )
        if not visual_documents:
            visual_documents = _load_visual_nodes(settings.pdf_dir)
        documents.extend(visual_documents)
    return [document for document in documents if _clean_text(document.page_content)]


def _to_llama_document(document: LangchainDocument) -> Document:
    metadata = _normalized_metadata(document.metadata)
    image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
    if image_path:
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        metadata["image_local_path"] = str(resolved.resolve())
        metadata["image_path"] = _repo_relative(resolved)
    return Document(text=_clean_text(document.page_content), metadata=metadata)


def write_document_cache(documents: Sequence[LangchainDocument], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [
            {
                "text": _clean_text(document.page_content),
                "metadata": _normalized_metadata(document.metadata),
            }
            for document in documents
        ]
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _vector_store(pinecone_index, namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace,
        batch_size=100,
        remove_text_from_metadata=False,
    )


def _clear_namespace(pinecone_index, namespace: str) -> None:
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        if "Namespace not found" not in str(exc):
            logger.warning("Could not clear Pinecone namespace %s: %s", namespace, exc)


def _build_namespace_index(
    namespace: str,
    documents: Sequence[LangchainDocument],
    pinecone_index,
    pinecone_index_name: str,
) -> Optional[dict]:
    if not documents:
        return None
    llama_documents = [_to_llama_document(document) for document in documents]
    vector_store = _vector_store(pinecone_index, namespace)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        llama_documents,
        storage_context=storage_context,
        show_progress=True,
        transformations=[],
    )
    stats = pinecone_index.describe_index_stats()
    namespace_stats = getattr(stats, "namespaces", None) or {}
    if hasattr(namespace_stats, "to_dict"):
        namespace_stats = namespace_stats.to_dict()
    vector_count = None
    if isinstance(namespace_stats, dict) and namespace in namespace_stats:
        vector_count = namespace_stats[namespace].get("vector_count")
    return {
        "namespace": namespace,
        "backend": "pinecone",
        "pinecone_index_name": pinecone_index_name,
        "document_count": len(documents),
        "vector_count": vector_count,
    }


def _build_manifest(
    settings: LlamaIndexPipelineSettings,
    documents: Sequence[LangchainDocument],
    namespace_payloads: Dict[str, dict],
) -> dict:
    csv_count = sum(1 for document in documents if str(document.metadata.get("source_type") or "").lower() == "csv")
    pdf_count = sum(1 for document in documents if str(document.metadata.get("retrieval_group") or "") == "pdf_text")
    visual_count = sum(1 for document in documents if str(document.metadata.get("retrieval_group") or "") == "visual")
    return {
        "storage_version": 3,
        "storage_root": str(settings.storage_dir),
        "document_cache_path": str(settings.document_cache_path),
        "vector_backend": "pinecone",
        "pinecone_index_name": settings.pinecone_index_name,
        "query_contract": {
            "structured_route": "csv_structured",
            "semantic_namespaces": [settings.pdf_namespace, settings.visual_namespace],
            "csv_rows_indexed": settings.index_csv_rows,
        },
        "namespaces": namespace_payloads,
        "counts": {
            "document_count": len(documents),
            "csv_count": csv_count,
            "pdf_text_count": pdf_count,
            "visual_count": visual_count,
        },
        "embedding_model": BGE_MODEL_NAME,
        "embedding_dimension": _current_embedding_dimension(),
    }


def build_llamaindex(settings: LlamaIndexPipelineSettings) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    documents = collect_source_documents(settings)
    if not documents:
        raise RuntimeError("No source documents were loaded for LlamaIndex ingestion.")

    if settings.rebuild_storage and settings.storage_dir.exists():
        shutil.rmtree(settings.storage_dir)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    write_document_cache(documents, settings.document_cache_path)

    Settings.embed_model = BgeLlamaIndexEmbedding(embed_batch_size=32)
    Settings.llm = None

    pinecone_index = _pinecone_index(settings.pinecone_index_name)

    namespace_payloads: Dict[str, dict] = {}
    pdf_documents = [
        document for document in documents if str(document.metadata.get("retrieval_group") or "") == "pdf_text"
    ]
    visual_documents = [
        document for document in documents if str(document.metadata.get("retrieval_group") or "") == "visual"
    ]
    csv_documents = [
        document for document in documents if str(document.metadata.get("source_type") or "").lower() == "csv"
    ]

    if settings.rebuild_storage:
        _clear_namespace(pinecone_index, settings.pdf_namespace)
        _clear_namespace(pinecone_index, settings.visual_namespace)
        if settings.index_csv_rows:
            _clear_namespace(pinecone_index, "csv_structured")

    pdf_payload = _build_namespace_index(settings.pdf_namespace, pdf_documents, pinecone_index, settings.pinecone_index_name)
    if pdf_payload:
        namespace_payloads[settings.pdf_namespace] = pdf_payload

    visual_payload = _build_namespace_index(settings.visual_namespace, visual_documents, pinecone_index, settings.pinecone_index_name)
    if visual_payload:
        namespace_payloads[settings.visual_namespace] = visual_payload

    if settings.index_csv_rows:
        csv_payload = _build_namespace_index("csv_structured", csv_documents, pinecone_index, settings.pinecone_index_name)
        if csv_payload:
            namespace_payloads["csv_structured"] = csv_payload

    manifest = _build_manifest(settings, documents, namespace_payloads)
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(
        logger,
        logging.INFO,
        "llamaindex_ingestion_completed",
        storage_root=str(settings.storage_dir),
        document_cache=str(settings.document_cache_path),
        manifest_path=str(settings.manifest_path),
        vector_backend="pinecone",
        pinecone_index_name=settings.pinecone_index_name,
        document_count=len(documents),
        indexed_document_count=sum(payload["document_count"] for payload in namespace_payloads.values()),
        csv_count=len(csv_documents),
        pdf_text_count=len(pdf_documents),
        visual_count=len(visual_documents),
        namespaces=list(namespace_payloads),
        csv_rows_indexed=settings.index_csv_rows,
    )
    return len(documents)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified LlamaIndex data brain.")
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR))
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--storage-dir", default=str(DEFAULT_STORAGE_DIR))
    parser.add_argument("--document-cache", default=str(DEFAULT_DOCUMENT_CACHE))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--visual-bootstrap-cache", default=str(DEFAULT_VISUAL_BOOTSTRAP_CACHE))
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-pdf-text", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--fresh-visual-extraction", action="store_true")
    parser.add_argument("--index-csv-rows", action="store_true")
    parser.add_argument("--keep-storage", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    count = build_llamaindex(
        LlamaIndexPipelineSettings(
            csv_dir=Path(args.csv_dir),
            pdf_dir=Path(args.pdf_dir),
            storage_dir=Path(args.storage_dir),
            document_cache_path=Path(args.document_cache),
            manifest_path=Path(args.manifest_path),
            visual_bootstrap_cache_path=Path(args.visual_bootstrap_cache),
            include_csv=not args.no_csv,
            include_pdf_text=not args.no_pdf_text,
            include_visuals=not args.no_visuals,
            bootstrap_visuals_from_cache=not args.fresh_visual_extraction,
            index_csv_rows=args.index_csv_rows,
            rebuild_storage=not args.keep_storage,
        )
    )
    print(f"Indexed {count} documents into Pinecone index {PINECONE_INDEX_NAME}")
