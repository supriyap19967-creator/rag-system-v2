import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    Pinecone = None
    ServerlessSpec = None

from app.embeddings import BGE_EMBEDDING_DIMENSIONS, BGE_MODEL_NAME, get_bge_embeddings
from app.ingestion import (
    DEFAULT_CSV_DIR,
    DEFAULT_PDF_DIR,
    infer_metric_family,
    load_ingestion_documents,
)
from app.utils import log_event

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 100
VISUAL_INDEX_SAMPLE_LIMIT = int(os.getenv("VISUAL_INDEX_SAMPLE_LIMIT", "5"))
VISUAL_INDEX_COUNT_LIMIT = int(os.getenv("VISUAL_INDEX_COUNT_LIMIT", "10000"))
PINECONE_METRIC = "cosine"
BM25_CACHE_PATH = Path(os.getenv("BM25_DOCUMENT_CACHE_PATH", "Data/bm25_documents.json"))
FIGURE_CAPTION_PATTERN = re.compile(
    r"\b((?:Figure|Fig\.?|Table|Chart|Panel)\s+\d+(?:\.\d+)?[A-Za-z]?\s*[:.\-]?\s*[^|]{12,280})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EmbeddingPipelineSettings:
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    recreate_index: bool = False

    @classmethod
    def from_env(cls) -> "EmbeddingPipelineSettings":
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()

        missing = [
            name
            for name, value in (
                ("PINECONE_API_KEY", pinecone_api_key),
                ("PINECONE_INDEX_NAME", pinecone_index_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "bge_small_v1").strip() or "bge_small_v1",
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip() or "aws",
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
            recreate_index=os.getenv("RECREATE_PINECONE_INDEX", "false").strip().lower()
            in {"1", "true", "yes"},
        )


def _batched(values: Sequence[Document], batch_size: int) -> Iterator[Sequence[Document]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return list(vector)
    return [component / norm for component in vector]


def _stable_vector_id(document: Document) -> str:
    metadata = document.metadata
    identity_parts = [
        str(metadata.get("source", metadata.get("source_type", "unknown"))),
        str(metadata.get("dataset_type", "")),
        str(metadata.get("row_index", "")),
        str(metadata.get("section_index", "")),
        str(metadata.get("chunk_index", "")),
        str(metadata.get("content_type", "")),
        str(metadata.get("image_path", "")),
        document.page_content,
    ]
    digest = hashlib.sha1("|".join(identity_parts).encode("utf-8")).hexdigest()
    return f"doc-{digest}"


def _metadata_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _response_value(response: object, key: str, default: object = None) -> object:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _matches_from_response(response: object) -> List[object]:
    if isinstance(response, dict):
        return list(response.get("matches") or [])
    return list(getattr(response, "matches", []) or [])


def _match_metadata(match: object) -> Dict[str, object]:
    if isinstance(match, dict):
        return dict(match.get("metadata") or {})
    return dict(getattr(match, "metadata", {}) or {})


def _namespace_vector_count(stats: object, namespace: str) -> Optional[int]:
    namespaces = _response_value(stats, "namespaces", {}) or {}
    if hasattr(namespaces, "to_dict"):
        namespaces = namespaces.to_dict()
    namespace_stats = namespaces.get(namespace) if isinstance(namespaces, dict) else None
    if namespace_stats is None:
        return None
    return _response_value(namespace_stats, "vector_count", None)


def _metadata_preview(metadata: Dict[str, object]) -> Dict[str, object]:
    image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
    return {
        "source_type": metadata.get("source_type"),
        "content_type": metadata.get("content_type"),
        "visual_type": metadata.get("visual_type"),
        "figure_id": metadata.get("figure_id"),
        "section": metadata.get("section") or metadata.get("section_header"),
        "source_pdf": metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source"),
        "page": metadata.get("source_page") or metadata.get("page"),
        "caption": str(metadata.get("caption") or "")[:240],
        "original_text_preview": str(metadata.get("original_text") or "")[:240],
        "image_path": image_path,
        "image_path_exists": bool(image_path and Path(image_path).exists()),
    }


def _document_country(document: Document) -> str:
    metadata = document.metadata
    if metadata.get("country_iso3"):
        return _metadata_value(metadata.get("country_iso3"))
    if metadata.get("country_codes"):
        return _metadata_value(metadata.get("country_codes"))
    return ""


def _caption_from_document(document: Document) -> str:
    metadata = document.metadata
    explicit_caption = _metadata_value(metadata.get("caption")).strip()
    if explicit_caption:
        return explicit_caption
    text = " ".join(
        _metadata_value(value)
        for value in (
            metadata.get("visual_data"),
            metadata.get("generated_description"),
            metadata.get("nearby_text"),
            document.page_content,
        )
    )
    match = FIGURE_CAPTION_PATTERN.search(text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" |")


def build_vector_metadata(document: Document) -> Dict[str, object]:
    metadata = document.metadata
    return {
        "original_text": document.page_content,
        "source": _metadata_value(metadata.get("source")),
        "source_files": _metadata_value(metadata.get("source_files")),
        "country": _document_country(document),
        "country_name": _metadata_value(metadata.get("country_name")),
        "country_iso3": _metadata_value(metadata.get("country_iso3")),
        "year": _metadata_value(metadata.get("year")),
        "dataset_type": _metadata_value(metadata.get("dataset_type")),
        "indicator": _metadata_value(metadata.get("indicator")),
        "metric_family": _metadata_value(
            metadata.get("metric_family")
            or infer_metric_family(metadata.get("indicator"), metadata.get("dataset_type"))
        ),
        "value": _metadata_value(metadata.get("value")),
        "row_index": _metadata_value(metadata.get("row_index")),
        "h1": _metadata_value(metadata.get("h1")),
        "h2": _metadata_value(metadata.get("h2")),
        "h3": _metadata_value(metadata.get("h3")),
        "source_type": _metadata_value(metadata.get("source_type")),
        "content_type": _metadata_value(metadata.get("content_type")),
        "element_type": _metadata_value(metadata.get("element_type")),
        "visual_type": _metadata_value(metadata.get("visual_type")),
        "figure_id": _metadata_value(metadata.get("figure_id")),
        "section": _metadata_value(metadata.get("section") or metadata.get("section_header")),
        "section_header": _metadata_value(metadata.get("section_header") or metadata.get("section")),
        "topic": _metadata_value(metadata.get("topic")),
        "page": _metadata_value(metadata.get("page")),
        "source_page": _metadata_value(metadata.get("source_page")),
        "image_path": _metadata_value(metadata.get("image_path")),
        "image_local_path": _metadata_value(metadata.get("image_local_path")),
        "is_multimodal": bool(metadata.get("is_multimodal")),
        "caption": _caption_from_document(document),
        "previous_text": _metadata_value(metadata.get("previous_text")),
        "next_text": _metadata_value(metadata.get("next_text")),
        "visual_data": _metadata_value(metadata.get("visual_data")),
        "nearby_text": _metadata_value(metadata.get("nearby_text")),
        "generated_description": _metadata_value(metadata.get("generated_description")),
        "vision_captioning_status": _metadata_value(metadata.get("vision_captioning_status")),
        "caption_source": _metadata_value(metadata.get("caption_source")),
    }


def write_bm25_document_cache(
    documents: Sequence[Document],
    cache_path: Path = BM25_CACHE_PATH,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [
            {
                "page_content": document.page_content,
                "metadata": build_vector_metadata(document),
            }
            for document in documents
            if str(document.page_content or "").strip()
        ]
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    log_event(
        logger,
        logging.INFO,
        "bm25_document_cache_written",
        cache_path=str(cache_path),
        document_count=len(payload["documents"]),
    )


class BgeEmbeddingService:
    def __init__(
        self,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._embeddings = get_bge_embeddings()
        self._batch_size = batch_size

    def embed_documents(self, documents: Sequence[Document]) -> List[List[float]]:
        embeddings: List[List[float]] = []

        for batch in _batched(documents, self._batch_size):
            texts = [document.page_content for document in batch]
            embeddings.extend(
                _l2_normalize(vector)
                for vector in self._embeddings.embed_documents(texts)
            )

        return embeddings


class PineconeVectorStoreService:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        cloud: str,
        region: str,
        namespace: str,
        recreate_index: bool,
        dimension: int = BGE_EMBEDDING_DIMENSIONS,
        metric: str = PINECONE_METRIC,
        upsert_batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        if Pinecone is None or ServerlessSpec is None:
            raise ImportError("pinecone is required for the embedding pipeline.")
        self._client = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._namespace = namespace
        self._dimension = dimension
        self._metric = metric
        self._upsert_batch_size = upsert_batch_size
        self._cloud = cloud
        self._region = region
        self._recreate_index = recreate_index
        self._ensure_index()
        self._index = self._client.Index(index_name)

    def _list_index_names(self) -> List[str]:
        listed = self._client.list_indexes()
        if hasattr(listed, "names"):
            return list(listed.names())
        if isinstance(listed, list):
            return [
                item.get("name", "")
                for item in listed
                if isinstance(item, dict) and item.get("name")
            ]
        return []

    def _ensure_index(self) -> None:
        index_names = self._list_index_names()
        log_event(
            logger,
            logging.INFO,
            "pinecone_pipeline_configuration",
            index=self._index_name,
            namespace=self._namespace,
            dimension=self._dimension,
            metric=self._metric,
            cloud=self._cloud,
            region=self._region,
            recreate_index=self._recreate_index,
        )
        if self._index_name in index_names:
            description = self._client.describe_index(self._index_name)
            existing_dimension = getattr(description, "dimension", None)
            existing_metric = getattr(description, "metric", None)
            log_event(
                logger,
                logging.INFO,
                "pinecone_existing_index",
                dimension=existing_dimension,
                metric=existing_metric,
                host=getattr(description, "host", None),
            )

            if (
                self._recreate_index
                or existing_dimension not in (None, self._dimension)
                or existing_metric not in (None, self._metric)
            ):
                logger.warning(
                    "Deleting Pinecone index '%s' before recreating it.",
                    self._index_name,
                )
                self._client.delete_index(self._index_name)
                while self._index_name in self._list_index_names():
                    time.sleep(2)
            else:
                return

        self._client.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._metric,
            spec=ServerlessSpec(
                cloud=self._cloud,
                region=self._region,
            ),
        )
        while self._index_name not in self._list_index_names():
            time.sleep(2)

    def upsert_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(documents) != len(embeddings):
            raise ValueError("Document count and embedding count must match.")

        vectors = [
            {
                "id": _stable_vector_id(document),
                "values": list(embedding),
                "metadata": build_vector_metadata(document),
            }
            for document, embedding in zip(documents, embeddings)
        ]

        total_upserted = 0
        for start in range(0, len(vectors), self._upsert_batch_size):
            batch = vectors[start : start + self._upsert_batch_size]
            print(f"--- Upserting batch to Pinecone: {start + 1}-{start + len(batch)} of {len(vectors)} ---", flush=True)
            upsert_response = self._index.upsert(vectors=batch, namespace=self._namespace)
            upserted_count = _response_value(upsert_response, "upserted_count", len(batch))
            try:
                total_upserted += int(upserted_count or 0)
            except (TypeError, ValueError):
                total_upserted += len(batch)
            log_event(
                logger,
                logging.INFO,
                "pinecone_upsert_batch",
                upserted_count=upserted_count,
                batch_size=len(batch),
                namespace=self._namespace,
            )
            
            res = self._index.describe_index_stats()
            log_event(
                logger,
                logging.INFO,
                "pinecone_index_stats",
                namespace=self._namespace,
                namespaces=_response_value(res, "namespaces", {}),
                total_vector_count=_response_value(res, "total_vector_count", None),
            )
        
        print("--- Successfully Indexed to Pinecone! ---", flush=True)
        return total_upserted

    def inspect_visual_documents(
        self,
        *,
        count_limit: int = VISUAL_INDEX_COUNT_LIMIT,
        sample_limit: int = VISUAL_INDEX_SAMPLE_LIMIT,
    ) -> Dict[str, object]:
        stats = self._index.describe_index_stats()
        total_vector_count = _response_value(stats, "total_vector_count", None)
        namespace_vector_count = _namespace_vector_count(stats, self._namespace)
        probe_vector = [0.0] * self._dimension
        probe_vector[0] = 1.0
        metadata_filter = {
            "$and": [
                {"source_type": {"$eq": "pdf"}},
                {"content_type": {"$eq": "visual"}},
            ]
        }
        response = self._index.query(
            namespace=self._namespace,
            vector=probe_vector,
            top_k=count_limit,
            include_metadata=True,
            filter=metadata_filter,
        )
        matches = _matches_from_response(response)
        samples = [_metadata_preview(_match_metadata(match)) for match in matches[:sample_limit]]
        visual_types: Dict[str, int] = {}
        pages = set()
        figure_ids = set()
        sections = set()
        schema_issues: Dict[str, int] = {}
        for match in matches:
            metadata = _match_metadata(match)
            visual_type = str(metadata.get("visual_type") or "<missing>").strip() or "<missing>"
            visual_types[visual_type] = visual_types.get(visual_type, 0) + 1
            page = str(metadata.get("source_page") or metadata.get("page") or "").strip()
            figure_id = str(metadata.get("figure_id") or "").strip()
            section = str(metadata.get("section") or metadata.get("section_header") or "").strip()
            if page:
                pages.add(page)
            if figure_id:
                figure_ids.add(figure_id)
            if section:
                sections.add(section)
            if metadata.get("source_type") != "pdf":
                schema_issues["source_type_not_pdf"] = schema_issues.get("source_type_not_pdf", 0) + 1
            if metadata.get("content_type") != "visual":
                schema_issues["content_type_not_visual"] = schema_issues.get("content_type_not_visual", 0) + 1
            if not metadata.get("visual_type"):
                schema_issues["missing_visual_type"] = schema_issues.get("missing_visual_type", 0) + 1
            if not metadata.get("figure_id"):
                schema_issues["missing_figure_id"] = schema_issues.get("missing_figure_id", 0) + 1
            if not (metadata.get("section") or metadata.get("section_header")):
                schema_issues["missing_section"] = schema_issues.get("missing_section", 0) + 1
        summary = {
            "index": self._index_name,
            "namespace": self._namespace,
            "total_vector_count": total_vector_count,
            "namespace_vector_count": namespace_vector_count,
            "visual_docs_count": len(matches),
            "visual_docs_count_limit": count_limit,
            "visual_count_truncated": len(matches) >= count_limit,
            "visual_types": visual_types,
            "unique_pages_covered": len(pages),
            "unique_figure_table_ids_covered": len(figure_ids),
            "unique_sections_covered": len(sections),
            "sample_pages": sorted(pages, key=lambda value: int(value) if value.isdigit() else value)[:20],
            "sample_figure_ids": sorted(figure_ids)[:20],
            "schema_issues": schema_issues,
            "samples": samples,
        }
        log_event(logger, logging.INFO, "pinecone_visual_index_verification", **summary)
        return summary


class EmbeddingPipeline:
    def __init__(self, settings: EmbeddingPipelineSettings) -> None:
        self._embedding_service = BgeEmbeddingService()
        self._vector_store_service = PineconeVectorStoreService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            namespace=settings.pinecone_namespace,
            recreate_index=settings.recreate_index,
        )

    def embed_and_store(self, documents: Sequence[Document]) -> None:
        csv_docs = [doc for doc in documents if doc.metadata.get("source_type") == "csv"]
        pdf_docs = [doc for doc in documents if doc.metadata.get("source_type") == "pdf"]
        visual_docs = [doc for doc in documents if doc.metadata.get("content_type") == "visual"]
        log_event(
            logger,
            logging.INFO,
            "embedding_pipeline_started",
            total_documents=len(documents),
            csv_documents=len(csv_docs),
            pdf_documents=len(pdf_docs),
            visual_documents=len(visual_docs),
            embedding_model=BGE_MODEL_NAME,
            embedding_dimensions=BGE_EMBEDDING_DIMENSIONS,
        )
        print(f"--- Embedding pipeline loaded {len(documents)} documents ({len(visual_docs)} visuals) ---", flush=True)
        
        embeddings = self._embedding_service.embed_documents(documents)
        if embeddings:
            log_event(
                logger,
                logging.INFO,
                "embedding_batch_completed",
                embedding_count=len(embeddings),
                first_embedding_dimension=len(embeddings[0]),
            )
        upserted_count = self._vector_store_service.upsert_documents(documents, embeddings)
        visual_summary = self._vector_store_service.inspect_visual_documents()
        log_event(
            logger,
            logging.INFO,
            "embedding_pipeline_visual_summary",
            extracted_visual_count=len(visual_docs),
            visual_chunks_created=len(visual_docs),
            visual_chunks_upserted=sum(1 for doc in documents if doc.metadata.get("content_type") == "visual"),
            total_vectors_upserted=upserted_count,
            pinecone_visual_count_after_upsert=visual_summary.get("visual_docs_count"),
            example_visuals=visual_summary.get("samples", [])[:3],
        )
        print("--- Visual Index Verification ---", flush=True)
        print(f"Extracted visual chunks: {len(visual_docs)}", flush=True)
        print(f"Total vectors upserted: {upserted_count}", flush=True)
        print(f"Pinecone visual docs after upsert: {visual_summary.get('visual_docs_count')}", flush=True)
        print(f"Unique visual pages covered: {visual_summary.get('unique_pages_covered')}", flush=True)
        print(f"Unique figure/table IDs covered: {visual_summary.get('unique_figure_table_ids_covered')}", flush=True)
        for sample in visual_summary.get("samples", [])[:3]:
            print(
                f"- page={sample.get('page')} id={sample.get('figure_id')} "
                f"type={sample.get('visual_type')} section={sample.get('section')} caption={sample.get('caption')}",
                flush=True,
            )
        logger.info("Successfully uploaded all vectors to Pinecone.")


def embed_and_store_ingestion_documents(
    pdf_dir: Optional[str] = None,
    csv_dir: Optional[str] = None,
    settings: Optional[EmbeddingPipelineSettings] = None,
) -> None:
    resolved_settings = settings or EmbeddingPipelineSettings.from_env()
    print("--- Starting embedding ingestion pipeline ---", flush=True)
    documents = load_ingestion_documents(
        csv_dir=DEFAULT_CSV_DIR if csv_dir is None else Path(csv_dir),
        pdf_dir=DEFAULT_PDF_DIR if pdf_dir is None else Path(pdf_dir),
        include_csv_vectors=False,
        include_pdf_visuals=True,
    )
    print(f"--- Writing BM25 cache for {len(documents)} documents ---", flush=True)
    write_bm25_document_cache(documents)
    pipeline = EmbeddingPipeline(resolved_settings)
    pipeline.embed_and_store(documents)


if __name__ == "__main__":
    embed_and_store_ingestion_documents()
