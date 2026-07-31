import os
import logging
import re
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None
from rank_bm25 import BM25Okapi

from app.embeddings import BGE_EMBEDDING_DIMENSIONS, get_bge_embeddings
from app.ingestion import (
    infer_metric_family,
)
from app.reranker import TransformersReranker
from app.utils import log_event


load_dotenv()

logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "bge_small_v1").strip() or "bge_small_v1"
PINECONE_TOP_K = 5
PINECONE_FETCH_K = 20
BM25_TOP_K = 5
RRF_K = 60
RERANK_FETCH_MULTIPLIER = 3
MIN_CLEAN_CHUNK_LENGTH = 50
QUALITY_FALLBACK_LIMIT = 2
BM25_CACHE_PATH = Path(os.getenv("BM25_DOCUMENT_CACHE_PATH", "Data/bm25_documents.json"))

QueryIntent = Literal["numerical", "explanatory", "mixed", "balanced"]


@dataclass(frozen=True)
class RetrievalResult:
    documents: List[Document]
    mode: str
    semantic_error: Optional[str] = None
    fallback_reason: Optional[str] = None
    semantic_match_count: int = 0
    bm25_match_count: int = 0
    metadata_filter: Optional[Dict[str, Any]] = None
    metadata_filter_relaxed: bool = False
    query_intent: Optional[str] = None

    @property
    def raw_context(self) -> str:
        return "\n".join(document.page_content for document in self.documents)


@dataclass(frozen=True)
class RetrievalHints:
    source_type: Optional[str] = None
    country_iso3: Optional[str] = None
    country_name: Optional[str] = None
    year: Optional[str] = None
    indicator_family: Optional[str] = None
    page: Optional[str] = None
    source_filename: Optional[str] = None
    topic: Optional[str] = None
    figure_id: Optional[str] = None
    visual_type: Optional[str] = None
    content_type: Optional[str] = None
    exact_only: bool = False

    def normalized(self) -> "RetrievalHints":
        return RetrievalHints(
            source_type=str(self.source_type or "").strip().lower() or None,
            country_iso3=str(self.country_iso3 or "").strip().upper() or None,
            country_name=str(self.country_name or "").strip() or None,
            year=str(self.year or "").strip() or None,
            indicator_family=str(self.indicator_family or "").strip().lower() or None,
            page=str(self.page or "").strip() or None,
            source_filename=str(self.source_filename or "").strip() or None,
            topic=str(self.topic or "").strip().lower() or None,
            figure_id=str(self.figure_id or "").strip() or None,
            visual_type=str(self.visual_type or "").strip().lower() or None,
            content_type=str(self.content_type or "").strip().lower() or None,
            exact_only=bool(self.exact_only),
        )


@dataclass
class _Bm25Index:
    documents: List[Document]
    model: Optional[BM25Okapi]
    tokenized_documents: List[List[str]]


QUALITY_KEYWORDS = {
    "standards",
    "development",
    "growth",
    "efficiency",
    "trade",
}
NOISY_PDF_KEYWORDS = {
    "contents",
    "references",
    "foreword",
    "figure",
    "table",
    "source:",
    "notes:",
}
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"[A-Z][^.!?]{25,}[.!?]")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_indicator_family(indicator: object, dataset_type: object = None) -> str:
    return infer_metric_family(indicator, dataset_type)


def _detect_query_intent(query: str) -> QueryIntent:
    normalized = query.lower()
    tokens = set(_tokenize(normalized))

    numerical_terms = {
        "gdp",
        "co2",
        "emissions",
        "growth",
        "value",
        "amount",
        "number",
        "numeric",
        "percent",
        "percentage",
        "capita",
        "usd",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    }
    explanatory_terms = {
        "explain",
        "why",
        "how",
        "affect",
        "impact",
        "procedure",
        "procedural",
        "process",
        "policy",
        "regulation",
        "regulations",
        "requirement",
        "requirements",
        "responsibilities",
        "standard",
        "standards",
        "guidelines",
        "definition",
        "describe",
        "report",
        "kyc",
        "cdd",
        "cip",
        "beneficial",
        "owner",
    }

    has_year_or_number = bool(re.search(r"\b(19|20)\d{2}\b|\d+(?:\.\d+)?", normalized))
    has_numerical = has_year_or_number or bool(tokens & numerical_terms)
    has_definition_shape = normalized.startswith(("what is ", "what are "))
    has_explanatory = bool(tokens & explanatory_terms) or (
        has_definition_shape and not has_numerical
    )

    if has_numerical and has_explanatory:
        return "mixed"
    if has_numerical:
        return "numerical"
    if has_explanatory:
        return "explanatory"
    return "balanced"


def _source_weights(intent: QueryIntent) -> Dict[str, float]:
    if intent == "numerical":
        return {"csv": 3.0, "pdf": 0.35, "unknown": 0.75}
    if intent == "explanatory":
        return {"csv": 0.65, "pdf": 2.0, "unknown": 0.9}
    if intent == "mixed":
        return {"csv": 1.2, "pdf": 1.2, "unknown": 1.0}
    return {"csv": 1.0, "pdf": 1.0, "unknown": 1.0}


def _source_type(document: Document) -> str:
    metadata = document.metadata
    raw_source_type = str(metadata.get("source_type", "")).strip().lower()
    if raw_source_type in {"csv", "pdf"}:
        return raw_source_type

    source = str(metadata.get("source") or metadata.get("source_files") or "").lower()
    if source.endswith(".csv") or "csv" in source:
        return "csv"
    if source.endswith(".pdf") or "pdf" in source:
        return "pdf"
    return "unknown"


def _metadata_filter_for_hints(
    hints: Optional[RetrievalHints],
    relaxed: bool = False,
) -> Optional[Dict[str, object]]:
    if hints is None:
        return None

    normalized = hints.normalized()
    clauses: List[Dict[str, object]] = []
    if normalized.source_type:
        clauses.append({"source_type": {"$eq": normalized.source_type}})
    if normalized.country_iso3:
        clauses.append({"country_iso3": {"$eq": normalized.country_iso3}})
    elif normalized.country_name:
        clauses.append({"country_name": {"$eq": normalized.country_name}})
    if normalized.year:
        clauses.append({"year": {"$eq": normalized.year}})
    if normalized.indicator_family and not relaxed:
        clauses.append({"metric_family": {"$eq": normalized.indicator_family}})
    if normalized.page and not relaxed:
        clauses.append({"page": {"$eq": normalized.page}})
    if normalized.source_filename and not relaxed:
        clauses.append({"source_files": {"$eq": normalized.source_filename}})
    if normalized.topic and not relaxed:
        clauses.append({"topic": {"$eq": normalized.topic}})
    if normalized.figure_id and not relaxed:
        clauses.append({"figure_id": {"$eq": normalized.figure_id}})
    if normalized.content_type:
        clauses.append({"content_type": {"$eq": normalized.content_type}})
    if normalized.visual_type and normalized.visual_type != "visual" and not relaxed:
        clauses.append({"visual_type": {"$eq": normalized.visual_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _document_source_weight(document: Document, weights: Dict[str, float]) -> float:
    return weights.get(_source_type(document), weights["unknown"])


def _clean_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sentence_count(text: str) -> int:
    return len(SENTENCE_PATTERN.findall(_clean_chunk_text(text)))


def _number_token_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)*", text)
    if not tokens:
        return 0.0
    number_tokens = sum(1 for token in tokens if re.search(r"\d", token))
    return number_tokens / len(tokens)


def _uppercase_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    uppercase_letters = sum(1 for character in letters if character.isupper())
    return uppercase_letters / len(letters)


def _has_meaningful_terms(text: str) -> bool:
    tokens = set(_tokenize(text))
    return bool(tokens & QUALITY_KEYWORDS)


def _looks_metadata_like(text: str) -> bool:
    stripped = _clean_chunk_text(text)
    if not stripped:
        return True

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines:
        short_lines = sum(1 for line in lines if len(line.split()) <= 4)
        if len(lines) >= 3 and short_lines / len(lines) >= 0.7:
            return True

    words = re.findall(r"[A-Za-z]+", stripped)
    punctuation_count = len(re.findall(r"[.!?]", stripped))
    if len(words) <= 12 and punctuation_count == 0:
        return True
    if _uppercase_ratio(stripped) > 0.6 and len(words) >= 5:
        return True
    return False


def _pdf_quality_rejection_reason(document: Document) -> Optional[str]:
    if str(document.metadata.get("content_type", "")).lower() == "visual":
        return None

    text = _clean_chunk_text(document.page_content)
    lowered = text.lower()
    sentence_count = _sentence_count(text)

    if len(text) < MIN_CLEAN_CHUNK_LENGTH:
        return "too_short"
    if URL_PATTERN.search(text):
        return "url"
    if any(keyword in lowered for keyword in ("source:", "notes:")):
        return "source_or_notes"
    if re.search(r"\b(?:figure|table)\s+\d+\b", lowered):
        return "figure_or_table"
    if re.search(r"\b(?:contents|references|foreword)\b", lowered) and sentence_count < 2:
        return "toc_or_references"
    if (
        "world development report" in lowered
        and sentence_count < 2
        and re.search(r"\bworld development report\s+\d{4}\s+\d+\b", lowered)
    ):
        return "report_page_header"
    if _number_token_ratio(text) > 0.3 and not _has_meaningful_terms(text):
        return "number_heavy"
    if _looks_metadata_like(text) and sentence_count == 0:
        return "metadata_like"
    return None


def _score_clean_pdf_chunk(document: Document) -> int:
    text = _clean_chunk_text(document.page_content)
    tokens = set(_tokenize(text))
    sentence_count = _sentence_count(text)
    score = 0

    score += min(sentence_count, 3) * 3
    score += min(len(tokens & QUALITY_KEYWORDS), 4) * 2
    if len(text) >= 180:
        score += 2
    if len(text) >= 320:
        score += 1
    if sentence_count < 2:
        score -= 2
    if _number_token_ratio(text) > 0.2:
        score -= 2
    if _looks_metadata_like(text):
        score -= 3
    return score


def _with_quality_metadata(document: Document, status: str, reason: Optional[str] = None) -> Document:
    metadata = dict(document.metadata)
    metadata["retrieval_quality_status"] = status
    if reason:
        metadata["retrieval_quality_reason"] = reason
    return Document(page_content=document.page_content, metadata=metadata)


def _filter_retrieved_documents(documents: Sequence[Document], top_k: int) -> List[Document]:
    clean_documents: List[Document] = []
    fallback_documents: List[Document] = []
    removed_reasons: Dict[str, int] = {}

    for document in documents:
        if str(document.metadata.get("content_type", "")).lower() == "visual":
            clean_documents.append(_with_quality_metadata(document, "kept_visual"))
            continue

        if _source_type(document) != "pdf":
            clean_documents.append(_with_quality_metadata(document, "kept_non_pdf"))
            continue

        rejection_reason = _pdf_quality_rejection_reason(document)
        if rejection_reason:
            removed_reasons[rejection_reason] = removed_reasons.get(rejection_reason, 0) + 1
            continue

        quality_score = _score_clean_pdf_chunk(document)
        enriched = _with_quality_metadata(document, "kept_clean")
        enriched.metadata["retrieval_quality_score"] = quality_score
        if quality_score >= 2:
            clean_documents.append(enriched)
        else:
            fallback_documents.append(_with_quality_metadata(document, "fallback_low_quality", "low_quality_score"))

    filtered = clean_documents[:top_k]
    if not filtered and fallback_documents:
        filtered = fallback_documents[: min(top_k, QUALITY_FALLBACK_LIMIT)]

    removed_count = len(documents) - len(filtered)
    log_event(
        logger,
        logging.INFO,
        "retrieval_quality_filter",
        input_chunks=len(documents),
        removed_chunks=removed_count,
        kept_chunks=len(filtered),
        removed_reasons=removed_reasons,
    )
    if not filtered and documents:
        logger.warning(
            "Retrieval quality filter removed all chunks; answer generation will receive no context."
        )
    return filtered


@lru_cache(maxsize=1)
def _load_bm25_index() -> _Bm25Index:
    started_at = time.monotonic()
    if not BM25_CACHE_PATH.exists():
        log_event(
            logger,
            logging.WARNING,
            "bm25_cache_missing",
            cache_path=str(BM25_CACHE_PATH),
            reason="live_query_will_not_run_ingestion",
            elapsed_seconds=round(time.monotonic() - started_at, 3),
        )
        return _Bm25Index(documents=[], model=None, tokenized_documents=[])

    try:
        with BM25_CACHE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        log_event(
            logger,
            logging.WARNING,
            "bm25_cache_load_failed",
            cache_path=str(BM25_CACHE_PATH),
            reason=str(exc),
            elapsed_seconds=round(time.monotonic() - started_at, 3),
        )
        return _Bm25Index(documents=[], model=None, tokenized_documents=[])

    documents: List[Document] = []
    for item in payload if isinstance(payload, list) else payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        page_content = str(item.get("page_content") or "").strip()
        if not page_content:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        documents.append(Document(page_content=page_content, metadata=dict(metadata)))

    tokenized_documents = [_tokenize(document.page_content) for document in documents]
    if not tokenized_documents:
        return _Bm25Index(documents=[], model=None, tokenized_documents=[])

    bm25_index = _Bm25Index(
        documents=documents,
        model=BM25Okapi(tokenized_documents),
        tokenized_documents=tokenized_documents,
    )
    log_event(
        logger,
        logging.INFO,
        "bm25_cache_loaded",
        cache_path=str(BM25_CACHE_PATH),
        document_count=len(documents),
        elapsed_seconds=round(time.monotonic() - started_at, 3),
    )
    return bm25_index


@lru_cache(maxsize=1)
def _pinecone_index():
    if Pinecone is None:
        raise ImportError("pinecone-client is not installed. Please add it to your environment or requirements.txt.")
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")
    log_event(
        logger,
        logging.INFO,
        "pinecone_connection_configured",
        index=PINECONE_INDEX_NAME,
        namespace=NAMESPACE,
        expected_dimension=BGE_EMBEDDING_DIMENSIONS,
        api_key_present=bool(PINECONE_API_KEY),
    )
    client = Pinecone(api_key=PINECONE_API_KEY)
    index_names = list(client.list_indexes().names())
    log_event(logger, logging.INFO, "pinecone_available_indexes", indexes=index_names)
    if PINECONE_INDEX_NAME not in index_names:
        raise RuntimeError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. "
            f"Available indexes: {index_names}"
        )

    description = client.describe_index(PINECONE_INDEX_NAME)
    dimension = getattr(description, "dimension", None)
    metric = getattr(description, "metric", None)
    log_event(
        logger,
        logging.INFO,
        "pinecone_index_description",
        dimension=dimension,
        metric=metric,
        host=getattr(description, "host", None),
    )
    if dimension not in (None, BGE_EMBEDDING_DIMENSIONS):
        raise RuntimeError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' has {dimension} dimensions; "
            f"expected {BGE_EMBEDDING_DIMENSIONS} for BGE."
        )
    return client.Index(PINECONE_INDEX_NAME)


def _describe_pinecone_stats(index: object) -> object:
    try:
        stats = index.describe_index_stats()
    except Exception as exc:
        logger.warning("Pinecone stats unavailable: %s", exc)
        return None

    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces")
    total_vector_count = getattr(stats, "total_vector_count", None)
    if total_vector_count is None and isinstance(stats, dict):
        total_vector_count = stats.get("total_vector_count")
    dimension = getattr(stats, "dimension", None)
    if dimension is None and isinstance(stats, dict):
        dimension = stats.get("dimension")

    log_event(
        logger,
        logging.INFO,
        "pinecone_index_stats",
        dimension=dimension,
        total_vector_count=total_vector_count,
        namespaces=namespaces,
    )
    return stats


@lru_cache(maxsize=1)
def _reranker() -> TransformersReranker:
    return TransformersReranker()


def _stable_document_key(document: Document) -> str:
    metadata = document.metadata
    return "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("row_index", "")),
            str(metadata.get("year", "")),
            document.page_content,
        ]
    )


def _pinecone_match_to_document(match: object, rank: int) -> Optional[Document]:
    metadata = dict(getattr(match, "metadata", {}) or {})
    text = str(metadata.get("original_text", "")).strip()
    if not text:
        return None

    metadata.update(
        {
            "retrieval_source": "pinecone",
            "semantic_rank": rank,
            "semantic_score": float(getattr(match, "score", 0.0) or 0.0),
            "embedding_dimensions": BGE_EMBEDDING_DIMENSIONS,
        }
    )
    metadata["metric_family"] = metadata.get("metric_family") or _normalize_indicator_family(
        metadata.get("indicator"), metadata.get("dataset_type")
    )
    return Document(page_content=text, metadata=metadata)


def _semantic_search(
    query: str,
    weights: Dict[str, float],
    hints: Optional[RetrievalHints] = None,
    top_k: int = PINECONE_TOP_K,
) -> Tuple[List[Document], Optional[Dict[str, object]], bool]:
    semantic_started_at = time.monotonic()
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    query_vector = get_bge_embeddings().embed_query(query)
    log_event(
        logger,
        logging.INFO,
        "semantic_query_embedding_generated",
        dimension=len(query_vector),
        expected_dimension=BGE_EMBEDDING_DIMENSIONS,
    )
    if len(query_vector) != BGE_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"BGE query embedding returned {len(query_vector)} dimensions; "
            f"expected {BGE_EMBEDDING_DIMENSIONS}."
        )

    index = _pinecone_index()
    _describe_pinecone_stats(index)

    filters_to_try: List[Tuple[Optional[Dict[str, object]], bool]] = []
    strict_filter = _metadata_filter_for_hints(hints, relaxed=False)
    relaxed_filter = _metadata_filter_for_hints(hints, relaxed=True)
    if strict_filter is not None:
        filters_to_try.append((strict_filter, False))
    if relaxed_filter is not None and relaxed_filter != strict_filter:
        filters_to_try.append((relaxed_filter, True))
    filters_to_try.append((None, False))

    attempted_filters: List[Optional[Dict[str, object]]] = []
    for metadata_filter, relaxed in filters_to_try:
        if metadata_filter in attempted_filters:
            continue
        attempted_filters.append(metadata_filter)
        log_event(
            logger,
            logging.INFO,
            "pinecone_query_started",
            index=PINECONE_INDEX_NAME,
            namespace=NAMESPACE,
            top_k=max(PINECONE_FETCH_K, top_k),
            metadata_filter=metadata_filter,
            metadata_filter_relaxed=relaxed,
        )
        query_kwargs: Dict[str, Any] = {
            "namespace": NAMESPACE,
            "vector": list(query_vector),
            "top_k": max(PINECONE_FETCH_K, top_k),
            "include_metadata": True,
        }
        if metadata_filter is not None:
            query_kwargs["filter"] = metadata_filter

        results = index.query(**query_kwargs)
        matches = getattr(results, "matches", []) or []
        log_event(
            logger,
            logging.INFO,
            "pinecone_query_completed",
            raw_match_count=len(matches),
            metadata_filter=metadata_filter,
            metadata_filter_relaxed=relaxed,
        )

        documents: List[Document] = []
        skipped_missing_text = 0
        for rank, match in enumerate(matches, start=1):
            document = _pinecone_match_to_document(match, rank)
            if document is not None:
                source_weight = _document_source_weight(document, weights)
                metadata = dict(document.metadata)
                metadata["source_type"] = _source_type(document)
                metadata["source_weight"] = source_weight
                metadata["weighted_semantic_score"] = (
                    float(metadata.get("semantic_score", 0.0)) * source_weight
                )
                document = Document(page_content=document.page_content, metadata=metadata)
                documents.append(document)
            else:
                skipped_missing_text += 1

        if skipped_missing_text:
            log_event(
                logger,
                logging.INFO,
                "pinecone_matches_skipped_missing_text",
                skipped_missing_text=skipped_missing_text,
            )
        documents.sort(
            key=lambda document: float(document.metadata.get("weighted_semantic_score", 0.0)),
            reverse=True,
        )
        semantic_documents = documents[:top_k]
        if semantic_documents:
            log_event(
                logger,
                logging.INFO,
                "semantic_retrieval_usable_documents",
                usable_documents=len(semantic_documents),
                metadata_filter=metadata_filter,
                metadata_filter_relaxed=relaxed,
                elapsed_seconds=round(time.monotonic() - semantic_started_at, 3),
            )
            return semantic_documents, metadata_filter, relaxed

    log_event(
        logger,
        logging.INFO,
        "semantic_retrieval_no_documents",
        elapsed_seconds=round(time.monotonic() - semantic_started_at, 3),
    )
    return [], strict_filter or relaxed_filter, bool(relaxed_filter and relaxed_filter != strict_filter)


def _bm25_search(query: str, top_k: int = BM25_TOP_K) -> List[Document]:
    started_at = time.monotonic()
    bm25_index = _load_bm25_index()
    if bm25_index.model is None or not bm25_index.documents:
        log_event(
            logger,
            logging.INFO,
            "bm25_search_skipped",
            reason="bm25_cache_empty_or_missing",
            cache_path=str(BM25_CACHE_PATH),
            elapsed_seconds=round(time.monotonic() - started_at, 3),
        )
        return []

    scores = bm25_index.model.get_scores(_tokenize(query))
    ranked: List[Tuple[int, float]] = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]

    documents: List[Document] = []
    for rank, (document_index, score) in enumerate(ranked, start=1):
        source_document = bm25_index.documents[document_index]
        metadata = dict(source_document.metadata)
        metadata.update(
            {
                "retrieval_source": "bm25",
                "bm25_rank": rank,
                "bm25_score": float(score),
            }
        )
        documents.append(
            Document(
                page_content=source_document.page_content,
                metadata=metadata,
            )
        )
    log_event(
        logger,
        logging.INFO,
        "bm25_search_completed",
        query=query,
        document_count=len(documents),
        elapsed_seconds=round(time.monotonic() - started_at, 3),
    )
    return documents


def _rrf_fuse(
    semantic_documents: Sequence[Document],
    bm25_documents: Sequence[Document],
    weights: Dict[str, float],
    intent: QueryIntent,
) -> List[Document]:
    scored: Dict[str, Tuple[float, Document]] = {}

    for source_name, documents in (
        ("pinecone", semantic_documents),
        ("bm25", bm25_documents),
    ):
        for rank, document in enumerate(documents, start=1):
            key = _stable_document_key(document)
            score, existing_document = scored.get(key, (0.0, document))
            merged_metadata = dict(existing_document.metadata)
            merged_metadata.update(document.metadata)
            sources = set(str(merged_metadata.get("retrieval_source", "")).split("+"))
            sources.discard("")
            sources.add(source_name)
            merged_metadata["retrieval_source"] = "+".join(sorted(sources))
            source_weight = _document_source_weight(document, weights)
            weighted_rrf_contribution = source_weight * (1.0 / (RRF_K + rank))
            merged_metadata["query_intent"] = intent
            merged_metadata["source_type"] = _source_type(document)
            merged_metadata["source_weight"] = source_weight
            merged_metadata["rrf_score"] = score + weighted_rrf_contribution
            scored[key] = (
                float(merged_metadata["rrf_score"]),
                Document(page_content=document.page_content, metadata=merged_metadata),
            )

    return [
        document
        for _score, document in sorted(
            scored.values(),
            key=lambda item: item[0],
            reverse=True,
        )
    ]


def _rank_bm25_fallback(
    bm25_documents: Sequence[Document],
    weights: Dict[str, float],
    intent: QueryIntent,
) -> List[Document]:
    weighted_documents: List[Document] = []
    for rank, document in enumerate(bm25_documents, start=1):
        metadata = dict(document.metadata)
        source_weight = _document_source_weight(document, weights)
        weighted_score = source_weight * (1.0 / (RRF_K + rank))
        metadata["query_intent"] = intent
        metadata["source_type"] = _source_type(document)
        metadata["source_weight"] = source_weight
        metadata["rrf_score"] = weighted_score
        weighted_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )

    return sorted(
        weighted_documents,
        key=lambda document: float(document.metadata.get("rrf_score", 0.0)),
        reverse=True,
    )


def _debug_document_metadata(label: str, document: Optional[Document]) -> None:
    if document is None:
        log_event(logger, logging.INFO, "retrieval_debug_document", label=label, document=None)
        return

    metadata = document.metadata
    log_event(
        logger,
        logging.INFO,
        "retrieval_debug_document",
        label=label,
        retrieval_source=metadata.get("retrieval_source"),
        source_type=metadata.get("source_type"),
        source=metadata.get("source") or metadata.get("source_files"),
        semantic_rank=metadata.get("semantic_rank"),
        semantic_score=metadata.get("semantic_score"),
        bm25_rank=metadata.get("bm25_rank"),
        bm25_score=metadata.get("bm25_score"),
        year=metadata.get("year"),
        indicator=metadata.get("indicator"),
        country=metadata.get("country_name") or metadata.get("country_iso3"),
        has_original_text=bool(document.page_content.strip()),
        text_preview=document.page_content[:160],
    )


def _document_debug_snapshot(documents: Sequence[Document], limit: int = 5) -> List[Dict[str, object]]:
    snapshot: List[Dict[str, object]] = []
    for document in documents[:limit]:
        metadata = document.metadata
        snapshot.append(
            {
                "source": metadata.get("source") or metadata.get("source_files"),
                "source_type": _source_type(document),
                "retrieval_source": metadata.get("retrieval_source"),
                "rrf_score": metadata.get("rrf_score"),
                "rerank_score": metadata.get("rerank_score"),
                "semantic_score": metadata.get("semantic_score"),
                "bm25_score": metadata.get("bm25_score"),
                "year": metadata.get("year"),
                "indicator": metadata.get("indicator"),
            }
        )
    return snapshot


def _exact_figure_documents(documents: Sequence[Document], figure_id: str) -> List[Document]:
    expected = str(figure_id or "").strip().lower()
    if not expected:
        return []
    exact_documents: List[Document] = []
    seen_keys = set()
    for document in documents:
        metadata = document.metadata
        actual = str(metadata.get("figure_id") or "").strip().lower()
        if actual != expected:
            continue
        key = (
            str(metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source") or ""),
            str(metadata.get("source_page") or metadata.get("page") or ""),
            actual,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        exact_documents.append(document)
    return exact_documents


def get_relevant_documents(
    query: str,
    top_k: int = 5,
    hints: Optional[RetrievalHints] = None,
) -> RetrievalResult:
    retrieval_started_at = time.monotonic()
    intent = _detect_query_intent(query)
    weights = _source_weights(intent)
    normalized_hints = hints.normalized() if hints is not None else None
    log_event(
        logger,
        logging.INFO,
        "retrieval_started",
        query=query,
        intent=intent,
        source_weights=weights,
        hints=normalized_hints.__dict__ if normalized_hints is not None else None,
    )

    semantic_documents: List[Document] = []
    semantic_error: Optional[str] = None
    fallback_reason: Optional[str] = None
    semantic_filter: Optional[Dict[str, object]] = None
    metadata_filter_relaxed = False
    try:
        semantic_documents, semantic_filter, metadata_filter_relaxed = _semantic_search(
            query,
            weights,
            hints=normalized_hints,
            top_k=PINECONE_TOP_K,
        )
    except Exception as exc:
        semantic_error = str(exc)
        fallback_reason = "pinecone_error"
        logger.warning("Semantic retrieval failed; falling back to BM25 only: %s", semantic_error)
    if semantic_error is None and not semantic_documents:
        semantic_error = "Pinecone returned no semantic matches."
        fallback_reason = "no_semantic_matches"
        logger.info("Semantic retrieval returned no usable matches; falling back to BM25 only.")

    if semantic_error is None and normalized_hints is not None and normalized_hints.figure_id:
        exact_documents = _exact_figure_documents(semantic_documents, normalized_hints.figure_id)
        if exact_documents:
            final_docs = _filter_retrieved_documents(exact_documents, top_k=top_k)
            mode = "semantic_exact_metadata"
            log_event(
                logger,
                logging.INFO,
                "anchored_visual_fast_path_used",
                query=query,
                figure_id=normalized_hints.figure_id,
                semantic_match_count=len(semantic_documents),
                exact_match_count=len(exact_documents),
                final_document_count=len(final_docs),
                metadata_filter=semantic_filter,
                metadata_filter_relaxed=metadata_filter_relaxed,
                skipped_bm25=True,
                skipped_reranker=True,
            )
            log_event(
                logger,
                logging.INFO,
                "retrieval_completed",
                mode=mode,
                final_document_count=len(final_docs),
                semantic_match_count=len(semantic_documents),
                bm25_match_count=0,
                fallback_reason=fallback_reason,
                metadata_filter=semantic_filter,
                metadata_filter_relaxed=metadata_filter_relaxed,
                elapsed_seconds=round(time.monotonic() - retrieval_started_at, 3),
            )
            return RetrievalResult(
                documents=final_docs,
                mode=mode,
                semantic_error=semantic_error,
                fallback_reason=fallback_reason,
                semantic_match_count=len(semantic_documents),
                bm25_match_count=0,
                metadata_filter=semantic_filter,
                metadata_filter_relaxed=metadata_filter_relaxed,
                query_intent=intent,
            )
        log_event(
            logger,
            logging.INFO,
            "anchored_visual_fast_path_missed",
            query=query,
            figure_id=normalized_hints.figure_id,
            semantic_match_count=len(semantic_documents),
            metadata_filter=semantic_filter,
            metadata_filter_relaxed=metadata_filter_relaxed,
        )

    bm25_documents = _bm25_search(query, top_k=BM25_TOP_K)
    log_event(
        logger,
        logging.INFO,
        "retrieval_candidate_counts",
        semantic_matches=len(semantic_documents),
        bm25_matches=len(bm25_documents),
        fallback_reason=fallback_reason,
        metadata_filter=semantic_filter,
        metadata_filter_relaxed=metadata_filter_relaxed,
    )
    _debug_document_metadata("Top semantic result metadata", semantic_documents[0] if semantic_documents else None)
    _debug_document_metadata("Top BM25 result metadata", bm25_documents[0] if bm25_documents else None)

    fusion_started_at = time.monotonic()
    if semantic_error is not None:
        candidates = _rank_bm25_fallback(bm25_documents, weights, intent)
        mode = "keyword_fallback"
    else:
        candidates = _rrf_fuse(semantic_documents, bm25_documents, weights, intent)
        mode = "hybrid"
    log_event(
        logger,
        logging.INFO,
        "retrieval_fusion_timing",
        query=query,
        candidate_count=len(candidates),
        elapsed_seconds=round(time.monotonic() - fusion_started_at, 3),
    )
    log_event(
        logger,
        logging.INFO,
        "retrieval_mode_selected",
        mode=mode,
        semantic_match_count=len(semantic_documents),
        bm25_match_count=len(bm25_documents),
        fallback_reason=fallback_reason,
    )

    rerank_fetch_k = max(top_k, min(len(candidates), top_k * RERANK_FETCH_MULTIPLIER))
    pre_rerank_snapshot = _document_debug_snapshot(candidates, limit=rerank_fetch_k)
    try:
        rerank_started_at = time.monotonic()
        reranked = _reranker().rerank(query, candidates, top_k=rerank_fetch_k)
        log_event(
            logger,
            logging.INFO,
            "reranker_stage_completed",
            query=query,
            pre_rerank_order=pre_rerank_snapshot,
            post_rerank_order=_document_debug_snapshot(reranked, limit=rerank_fetch_k),
            elapsed_seconds=round(time.monotonic() - rerank_started_at, 3),
        )
    except Exception as exc:
        logger.warning("Reranking failed; returning fused retrieval order: %s", exc)
        reranked = list(candidates[:rerank_fetch_k])
        log_event(
            logger,
            logging.WARNING,
            "reranker_stage_failed",
            query=query,
            reason=str(exc),
            pre_rerank_order=pre_rerank_snapshot,
        )

    quality_filter_started_at = time.monotonic()
    final_docs = _filter_retrieved_documents(reranked, top_k=top_k)
    log_event(
        logger,
        logging.INFO,
        "retrieval_quality_filter_timing",
        query=query,
        elapsed_seconds=round(time.monotonic() - quality_filter_started_at, 3),
        final_document_count=len(final_docs),
    )
    if not final_docs and bm25_documents and mode == "hybrid":
        fallback_reason = "quality_filter_removed_semantic_candidates"
        mode = "keyword_fallback"
        final_docs = _filter_retrieved_documents(
            _rank_bm25_fallback(bm25_documents, weights, intent),
            top_k=top_k,
        )
        log_event(
            logger,
            logging.WARNING,
            "retrieval_mode_changed_after_quality_filter",
            mode=mode,
            fallback_reason=fallback_reason,
        )

    log_event(
        logger,
        logging.INFO,
        "retrieval_completed",
        mode=mode,
        final_document_count=len(final_docs),
        semantic_match_count=len(semantic_documents),
        bm25_match_count=len(bm25_documents),
        fallback_reason=fallback_reason,
        metadata_filter=semantic_filter,
        metadata_filter_relaxed=metadata_filter_relaxed,
        elapsed_seconds=round(time.monotonic() - retrieval_started_at, 3),
    )

    return RetrievalResult(
        documents=final_docs,
        mode=mode,
        semantic_error=semantic_error,
        fallback_reason=fallback_reason,
        semantic_match_count=len(semantic_documents),
        bm25_match_count=len(bm25_documents),
        metadata_filter=semantic_filter,
        metadata_filter_relaxed=metadata_filter_relaxed,
        query_intent=intent,
    )
