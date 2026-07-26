import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.pinecone import PineconeVectorStore

from app.llamaindex_embedding import BgeLlamaIndexEmbedding
from app.llamaindex_pipeline import (
    DEFAULT_DOCUMENT_CACHE,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_STORAGE_DIR,
    PDF_NAMESPACE,
    REPO_ROOT,
    VISUAL_NAMESPACE,
    _pinecone_index,
)
from app.llm import get_hybrid_llm
from app.schemas import SourceCitation
from app.utils import log_event


load_dotenv()

logger = logging.getLogger(__name__)
LLAMAINDEX_TOP_K = int(os.getenv("LLAMAINDEX_TOP_K", "8"))
LLAMAINDEX_VISUAL_TOP_K = int(os.getenv("LLAMAINDEX_VISUAL_TOP_K", "6"))
INSUFFICIENT_DATA_MESSAGE = "I do not have sufficient data to answer this question."
FIGURE_ID_PATTERN = re.compile(r"\b(Figure|Fig\.?|Table|Chart)\s+(\d+(?:\.\d+)?[A-Za-z]?)\b", re.IGNORECASE)
COUNTRY_QUERY_ALIASES = {
    "india": {"india", "ind"},
    "ind": {"india", "ind"},
    "united states": {"united states", "united states of america", "usa", "us"},
    "united states of america": {"united states", "united states of america", "usa", "us"},
    "usa": {"united states", "united states of america", "usa", "us"},
    "us": {"united states", "united states of america", "usa", "us"},
}


@dataclass(frozen=True)
class QueryBuild:
    route: str
    namespace_order: List[str]
    figure_id: Optional[str]
    figure_ids: List[str]
    visual_kind: Optional[str]
    year: Optional[str]
    metric_family: str
    country_aliases: set[str]
    query_tokens: List[str]
    wants_visual_attachment: bool


@dataclass(frozen=True)
class LlamaIndexAnswer:
    answer: str
    confidence_score: float
    source_citations: List[SourceCitation]
    sources: List[str]
    contexts: List[str]
    retrieved_chunks: List[dict]
    visual_results: List[dict]
    debug_info: Dict[str, object]
    model_used: str = "llamaindex"
    retrieval_mode: str = "llamaindex"
    fallback_reasons: List[str] = None


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _resolve_local_path(path: object) -> str:
    raw_path = str(path or "").strip()
    if not raw_path:
        return ""
    local_path = Path(raw_path)
    if not local_path.is_absolute():
        local_path = REPO_ROOT / local_path
    return str(local_path.resolve())


def _source_label(metadata: Dict[str, object]) -> str:
    return str(metadata.get("source_files") or metadata.get("source") or "unknown")


def _page_number(metadata: Dict[str, object]) -> Optional[int]:
    raw = metadata.get("source_page") or metadata.get("page")
    try:
        return int(raw) if raw not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _node_to_document(node: NodeWithScore, namespace: str) -> LangchainDocument:
    source_node = node.node
    metadata = dict(source_node.metadata or {})
    metadata["rerank_score"] = float(node.score or 0.0)
    metadata["retrieval_namespace"] = namespace
    return LangchainDocument(page_content=source_node.get_content(metadata_mode="none"), metadata=metadata)


def _chunk_payload(document: LangchainDocument) -> dict:
    metadata = document.metadata
    return {
        "text": document.page_content,
        "filename": str(metadata.get("source") or metadata.get("source_files") or "unknown"),
        "rerank_score": float(metadata.get("rerank_score", 0.0) or 0.0),
        "source_type": str(metadata.get("source_type") or ""),
        "content_type": str(metadata.get("content_type") or ""),
        "visual_type": str(metadata.get("visual_type") or ""),
        "figure_id": str(metadata.get("figure_id") or ""),
        "image_path": str(metadata.get("image_path") or metadata.get("image_local_path") or ""),
        "image_local_path": str(metadata.get("image_local_path") or metadata.get("image_path") or ""),
        "page_number": metadata.get("source_page") or metadata.get("page"),
        "retrieval_namespace": str(metadata.get("retrieval_namespace") or metadata.get("retrieval_group") or ""),
    }


def _visual_payload(document: LangchainDocument) -> Optional[dict]:
    metadata = document.metadata
    if str(metadata.get("content_type") or "").lower() != "visual":
        return None
    image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
    resolved_path = _resolve_local_path(image_path)
    if not resolved_path or not Path(resolved_path).exists():
        return None
    caption = str(metadata.get("caption") or metadata.get("generated_description") or document.page_content).strip()
    return {
        "image_path": str(metadata.get("image_path") or image_path),
        "image_local_path": resolved_path,
        "source_pdf": _source_label(metadata),
        "page_number": metadata.get("source_page") or metadata.get("page"),
        "visual_type": str(metadata.get("visual_type") or "visual"),
        "figure_id": str(metadata.get("figure_id") or ""),
        "caption": caption,
        "description": caption,
        "visual_relevance_score": float(metadata.get("rerank_score", 0.0) or 0.0),
        "image_path_exists": True,
    }


def _dedupe_documents(documents: Sequence[LangchainDocument]) -> List[LangchainDocument]:
    deduped: List[LangchainDocument] = []
    seen = set()
    for document in documents:
        metadata = document.metadata
        key = (
            str(metadata.get("source") or metadata.get("source_files") or ""),
            str(metadata.get("page") or metadata.get("source_page") or ""),
            str(metadata.get("figure_id") or ""),
            str(metadata.get("row_index") or metadata.get("chunk_index") or ""),
            document.page_content[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def _citation_list(documents: Sequence[LangchainDocument]) -> List[SourceCitation]:
    citations: List[SourceCitation] = []
    seen = set()
    for document in documents[:5]:
        citation = SourceCitation(filename=_source_label(document.metadata), page_number=_page_number(document.metadata))
        key = (citation.filename, citation.page_number)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def _sources(documents: Sequence[LangchainDocument]) -> List[str]:
    return sorted({_source_label(document.metadata) for document in documents if _source_label(document.metadata)})


def _requested_visual_kind(question: str) -> Optional[str]:
    normalized = _normalize(question)
    if re.search(r"\b(table|tabel)\b", normalized):
        return "table"
    if re.search(r"\b(chart|graph|plot|figure|fig|diagram|visual|image|show)\b", normalized):
        return "visual"
    return None


def _requested_figure_id(question: str) -> Optional[str]:
    match = FIGURE_ID_PATTERN.search(question or "")
    if not match:
        return None
    kind, number = match.groups()
    if kind.lower().startswith("fig"):
        kind = "Figure"
    else:
        kind = kind.title()
    return f"{kind} {number}"


def _requested_year(question: str) -> Optional[str]:
    match = re.search(r"\b(19|20)\d{2}\b", question)
    return match.group(0) if match else None


def _requested_metric_family(question: str) -> str:
    normalized = _normalize(question)
    if "gdp" in normalized:
        return "gdp"
    if "co2" in normalized or "emission" in normalized or "emissions" in normalized:
        return "co2"
    return ""


def _is_exact_numeric_question(question: str) -> bool:
    return bool(_requested_year(question) and _requested_metric_family(question))


def _query_country_aliases(question: str) -> set[str]:
    normalized = _normalize(question)
    aliases: set[str] = set()
    for alias, values in COUNTRY_QUERY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            aliases.update(values)
    tokens = set(normalized.split())
    aliases.update(token for token in tokens if len(token) == 3)
    return aliases


def _country_matches_query(metadata: Dict[str, object], query_aliases: set[str]) -> bool:
    if not query_aliases:
        return True
    country_name = _normalize(metadata.get("country_name"))
    country_code = _normalize(metadata.get("country_code") or metadata.get("country_iso3"))
    if country_name in query_aliases or country_code in query_aliases:
        return True
    return any(alias in country_name.split() for alias in query_aliases)


def _important_query_tokens(question: str) -> List[str]:
    stopwords = {
        "show",
        "give",
        "find",
        "about",
        "the",
        "and",
        "for",
        "with",
        "visual",
        "figure",
        "fig",
        "table",
        "tabel",
        "chart",
        "diagram",
        "graph",
        "what",
        "does",
        "say",
    }
    tokens = [token for token in _normalize(question).split() if len(token) >= 3 and token not in stopwords]
    expanded: List[str] = []
    for token in tokens:
        expanded.append(token)
        if token in {"adoption", "adopting", "adopted"}:
            expanded.append("adopt")
        if token.endswith("s") and len(token) > 4:
            expanded.append(token[:-1])
    return sorted(set(expanded))


def _is_short_query(question: str) -> bool:
    tokens = _normalize(question).split()
    return len(tokens) <= 5


def build_query(question: str) -> QueryBuild:
    # Use FIGURE_ID_PATTERN to find all figure IDs
    matches = FIGURE_ID_PATTERN.findall(question or "")
    figure_ids = []
    for kind, number in matches:
        if kind.lower().startswith("fig"):
            kind = "Figure"
        else:
            kind = kind.title()
        figure_ids.append(f"{kind} {number}")
    
    figure_id = figure_ids[0] if figure_ids else None
    visual_kind = _requested_visual_kind(question)
    year = _requested_year(question)
    metric_family = _requested_metric_family(question)
    country_aliases = _query_country_aliases(question)
    query_tokens = _important_query_tokens(question)
    wants_visual_attachment = bool(
        visual_kind or re.search(r"\b(show|chart|graph|figure|fig|table|diagram|image)\b", question, re.IGNORECASE)
    )

    if _is_exact_numeric_question(question):
        route = "structured_exact"
        namespace_order: List[str] = []
    elif figure_ids:
        route = "visual_exact"
        namespace_order = [VISUAL_NAMESPACE]
    elif visual_kind:
        route = "visual_semantic"
        namespace_order = [VISUAL_NAMESPACE]
    elif _is_short_query(question):
        route = "hybrid_short"
        namespace_order = [VISUAL_NAMESPACE, PDF_NAMESPACE]
    else:
        route = "text_semantic"
        namespace_order = [PDF_NAMESPACE]

    return QueryBuild(
        route=route,
        namespace_order=namespace_order,
        figure_id=figure_id,
        figure_ids=figure_ids,
        visual_kind=visual_kind,
        year=year,
        metric_family=metric_family,
        country_aliases=country_aliases,
        query_tokens=query_tokens,
        wants_visual_attachment=wants_visual_attachment,
    )


def _load_document_cache(cache_path: Path = DEFAULT_DOCUMENT_CACHE) -> List[LangchainDocument]:
    if not cache_path.exists():
        return []
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    documents: List[LangchainDocument] = []
    for item in payload.get("documents", []):
        documents.append(
            LangchainDocument(
                page_content=str(item.get("text") or item.get("page_content") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return documents


def _image_area(metadata: Dict[str, object]) -> int:
    image_path = _resolve_local_path(metadata.get("image_local_path") or metadata.get("image_path"))
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            return int(width) * int(height)
    except Exception:
        return 0


def _visual_document_quality_score(document: LangchainDocument) -> float:
    metadata = document.metadata
    caption = str(metadata.get("caption") or document.page_content or "")
    normalized_caption = _normalize(caption)
    score = 0.0
    score += min(_image_area(metadata) / 10000.0, 80.0)
    if str(metadata.get("image_local_path") or metadata.get("image_path") or "") and Path(
        _resolve_local_path(metadata.get("image_local_path") or metadata.get("image_path"))
    ).exists():
        score += 25.0
    if re.search(r"\b(main|summary|compliance|recommendations|typology|evidence)\b", normalized_caption):
        score += 12.0
    if re.search(r"\bpanel\b|\byields\b|\bchapter\b|\bthis is\b|\bbased on\b|\bcannot convert\b", normalized_caption):
        score -= 25.0
    if ")." in caption or caption.strip().endswith(("com-", "sev-", "âƒ")):
        score -= 18.0
    return score


def _visual_topic_score(document: LangchainDocument, query_tokens: Sequence[str]) -> float:
    metadata = document.metadata
    searchable = _normalize(
        " ".join(
            str(value or "")
            for value in (
                metadata.get("caption"),
                metadata.get("figure_id"),
                document.page_content,
            )
        )
    )
    score = _visual_document_quality_score(document) * 0.05
    for token in query_tokens:
        if re.search(rf"\b{re.escape(token)}\w*\b", searchable):
            score += 10.0
    if not query_tokens:
        score += 1.0
    return score


def _generic_keyword_score(document: LangchainDocument, query_tokens: Sequence[str]) -> float:
    metadata = document.metadata
    searchable = _normalize(
        " ".join(
            str(value or "")
            for value in (
                metadata.get("caption"),
                metadata.get("figure_id"),
                metadata.get("section"),
                metadata.get("section_header"),
                document.page_content,
            )
        )
    )
    score = float(metadata.get("rerank_score", 0.0) or 0.0)
    for token in query_tokens:
        if re.search(rf"\b{re.escape(token)}\w*\b", searchable):
            score += 3.5
    return score


def _exact_csv_documents(question: str, documents: Sequence[LangchainDocument]) -> Optional[List[LangchainDocument]]:
    if not _is_exact_numeric_question(question):
        return None
    year = _requested_year(question)
    metric_family = _requested_metric_family(question)
    country_aliases = _query_country_aliases(question)
    matches: List[LangchainDocument] = []
    for document in documents:
        metadata = document.metadata
        if str(metadata.get("source_type") or "").lower() != "csv":
            continue
        if str(metadata.get("year") or "") != year:
            continue
        if metric_family and str(metadata.get("metric_family") or "").lower() != metric_family:
            continue
        if not _country_matches_query(metadata, country_aliases):
            continue
        matches.append(document)
    return matches[:5]


def _direct_visual_matches(query: QueryBuild, cache_documents: Sequence[LangchainDocument]) -> List[LangchainDocument]:
    # 1. Multiple specific figure/table IDs
    if query.figure_ids:
        results = []
        for fid in query.figure_ids:
            matches = [
                doc for doc in cache_documents
                if str(doc.metadata.get("content_type") or "").lower() == "visual"
                and str(doc.metadata.get("figure_id") or "") == fid
            ]
            if matches:
                best_match = sorted(matches, key=_visual_document_quality_score, reverse=True)[0]
                results.append(best_match)
        seen = set()
        unique_results = []
        for doc in results:
            fid = doc.metadata.get("figure_id")
            if fid not in seen:
                seen.add(fid)
                unique_results.append(doc)
        if unique_results:
            return unique_results

    # 2. Generic request asking for multiple (e.g. "show the figure and table" or "show the charts")
    normalized_q = " ".join(query.query_tokens).lower()
    wants_both = ("table" in normalized_q or "tabel" in normalized_q) and \
                 any(w in normalized_q for w in ["figure", "fig", "chart", "graph", "diagram", "image", "visual"])
    
    if wants_both:
        tables = [
            doc for doc in cache_documents
            if str(doc.metadata.get("content_type") or "").lower() == "visual"
            and str(doc.metadata.get("visual_type") or "").lower() == "table"
        ]
        figures = [
            doc for doc in cache_documents
            if str(doc.metadata.get("content_type") or "").lower() == "visual"
            and str(doc.metadata.get("visual_type") or "").lower() == "figure"
        ]
        
        ranked_tables = sorted(tables, key=lambda d: _visual_topic_score(d, query.query_tokens), reverse=True)
        ranked_figures = sorted(figures, key=lambda d: _visual_topic_score(d, query.query_tokens), reverse=True)
        
        results = []
        if ranked_tables and _visual_topic_score(ranked_tables[0], query.query_tokens) > 0:
            results.append(ranked_tables[0])
        if ranked_figures and _visual_topic_score(ranked_figures[0], query.query_tokens) > 0:
            results.append(ranked_figures[0])
        if results:
            return results

    # 3. Single visual kind query fallback
    if query.visual_kind:
        matches = [
            document
            for document in cache_documents
            if str(document.metadata.get("content_type") or "").lower() == "visual"
            and (
                query.visual_kind != "table"
                or str(document.metadata.get("visual_type") or "").lower() == "table"
            )
        ]
        ranked = sorted(matches, key=lambda document: _visual_topic_score(document, query.query_tokens), reverse=True)
        return [document for document in ranked if _visual_topic_score(document, query.query_tokens) > 0][:4]

    return []


def _attach_related_visuals(
    question: str,
    documents: Sequence[LangchainDocument],
    cache_documents: Sequence[LangchainDocument],
) -> List[LangchainDocument]:
    if not documents:
        return []
    query_tokens = _important_query_tokens(question)
    candidates = [
        document
        for document in cache_documents
        if str(document.metadata.get("content_type") or "").lower() == "visual"
    ]
    ranked = sorted(candidates, key=lambda document: _visual_topic_score(document, query_tokens), reverse=True)
    attachments: List[LangchainDocument] = []
    for document in ranked:
        if _visual_topic_score(document, query_tokens) <= 0:
            continue
        attachments.append(document)
        if len(attachments) >= 1:
            break
    return attachments


@lru_cache(maxsize=1)
def _load_manifest(manifest_path: str) -> dict:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _namespace_index(index_name: str, namespace: str) -> VectorStoreIndex:
    Settings.embed_model = BgeLlamaIndexEmbedding(embed_batch_size=32)
    Settings.llm = None
    pinecone_index = _pinecone_index(index_name)
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace,
        remove_text_from_metadata=False,
    )
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)


def _namespace_configs(manifest_path: Path) -> Dict[str, dict]:
    manifest = _load_manifest(str(manifest_path))
    namespaces = manifest.get("namespaces") or {}
    if namespaces:
        return {str(namespace): dict(payload or {}) for namespace, payload in namespaces.items()}
    return {}


def _retrieve_namespace(question: str, namespace: str, index_name: str, top_k: int) -> List[LangchainDocument]:
    index = _namespace_index(index_name, namespace)
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(question)
    return [_node_to_document(node, namespace) for node in nodes]


def _retrieve(question: str, query: QueryBuild, manifest_path: Path, cache_documents: Sequence[LangchainDocument]) -> List[LangchainDocument]:
    exact_csv_matches = _exact_csv_documents(question, cache_documents)
    if exact_csv_matches is not None:
        return exact_csv_matches

    namespace_configs = _namespace_configs(manifest_path)
    direct_visual = _direct_visual_matches(query, cache_documents)

    def namespace_index_name(namespace: str) -> str:
        payload = namespace_configs.get(namespace) or {}
        configured = str(payload.get("pinecone_index_name") or "").strip()
        if configured and not configured.startswith("http"):
            return configured
        return str(os.getenv("PINECONE_INDEX_NAME") or "").strip()

    if query.route == "visual_exact":
        if direct_visual:
            return direct_visual
        if VISUAL_NAMESPACE in namespace_configs:
            return _dedupe_documents(
                _retrieve_namespace(question, VISUAL_NAMESPACE, namespace_index_name(VISUAL_NAMESPACE), LLAMAINDEX_VISUAL_TOP_K)
            )
        return []

    if query.route == "visual_semantic":
        documents = list(direct_visual)
        if VISUAL_NAMESPACE in namespace_configs:
            documents.extend(
                _retrieve_namespace(question, VISUAL_NAMESPACE, namespace_index_name(VISUAL_NAMESPACE), LLAMAINDEX_VISUAL_TOP_K)
            )
        ranked = sorted(_dedupe_documents(documents), key=lambda document: _visual_topic_score(document, query.query_tokens), reverse=True)
        return ranked[:LLAMAINDEX_VISUAL_TOP_K]

    if query.route == "hybrid_short":
        documents = list(direct_visual)
        for namespace in query.namespace_order:
            if namespace not in namespace_configs:
                continue
            documents.extend(
                _retrieve_namespace(
                    question,
                    namespace,
                    namespace_index_name(namespace),
                    LLAMAINDEX_VISUAL_TOP_K if namespace == VISUAL_NAMESPACE else LLAMAINDEX_TOP_K,
                )
            )
        ranked = sorted(_dedupe_documents(documents), key=lambda document: _generic_keyword_score(document, query.query_tokens), reverse=True)
        return ranked[:LLAMAINDEX_TOP_K]

    documents: List[LangchainDocument] = []
    if PDF_NAMESPACE in namespace_configs:
        documents.extend(_retrieve_namespace(question, PDF_NAMESPACE, namespace_index_name(PDF_NAMESPACE), LLAMAINDEX_TOP_K))
    if query.wants_visual_attachment:
        documents.extend(_attach_related_visuals(question, documents, cache_documents))
    return _dedupe_documents(documents)


def _deterministic_answer(question: str, documents: Sequence[LangchainDocument], visual_results: Sequence[dict]) -> tuple[str, float, str]:
    if visual_results:
        labels = []
        for visual in visual_results:
            label = str(visual.get("figure_id") or "").strip()
            caption = str(visual.get("caption") or "").strip()
            labels.append(caption if caption else label)
        return "Showing: " + "; ".join(label for label in labels if label), 0.82, "llamaindex-visual"

    if not documents:
        return INSUFFICIENT_DATA_MESSAGE, 0.0, "llamaindex-empty"

    exact_csv_answer = _exact_csv_answer(question, documents)
    if exact_csv_answer:
        return exact_csv_answer, 0.9, "llamaindex-csv"

    evidence = "\n\n".join(
        f"[{index}] {document.page_content}"
        for index, document in enumerate(documents[:4], start=1)
    )
    llm = get_hybrid_llm()
    if llm.is_available():
        prompt = (
            "Answer the question using only the evidence below. "
            f"If the evidence is insufficient, say exactly: {INSUFFICIENT_DATA_MESSAGE}\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        result = llm.invoke(prompt, session_id=None)
        answer = str(result.get("answer") or "").strip()
        lowered = answer.lower()
        if answer and not lowered.startswith("llm error") and not lowered.startswith("llm unavailable"):
            return answer, 0.74, str(result.get("model_used") or "llamaindex-llm")

    snippets = [document.page_content.strip() for document in documents[:3] if document.page_content.strip()]
    return "\n\n".join(snippets) if snippets else INSUFFICIENT_DATA_MESSAGE, 0.55, "llamaindex-local"


def _exact_csv_answer(question: str, documents: Sequence[LangchainDocument]) -> str:
    year = _requested_year(question)
    if not year:
        return ""
    for document in documents:
        metadata = document.metadata
        if str(metadata.get("source_type") or "").lower() != "csv":
            continue
        if str(metadata.get("year") or "") != year:
            continue
        country = str(metadata.get("country_name") or "")
        indicator = str(metadata.get("indicator") or "")
        value = str(metadata.get("value") or "").strip()
        if not value:
            match = re.search(rf"\b{re.escape(year)}:\s*([^;\n]+)", document.page_content)
            value = match.group(1).strip() if match else ""
        if not value:
            continue
        return f"In {year}, {indicator} for {country} was {value}."
    return ""


def query_llamaindex(
    *,
    question: str,
    session_id: str,
    include_debug: bool = False,
    storage_dir: Path = DEFAULT_STORAGE_DIR,
    document_cache_path: Path = DEFAULT_DOCUMENT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> LlamaIndexAnswer:
    namespace_configs = _namespace_configs(manifest_path)
    if not namespace_configs:
        return LlamaIndexAnswer(
            answer=f"LlamaIndex Pinecone manifest was not found at {manifest_path}. Run: python app\\llamaindex_pipeline.py",
            confidence_score=0.0,
            source_citations=[],
            sources=[],
            contexts=[],
            retrieved_chunks=[],
            visual_results=[],
            debug_info={"storage_missing": str(storage_dir)} if include_debug else {},
            model_used="llamaindex-not-built",
            fallback_reasons=["llamaindex_manifest_missing"],
        )

    cache_documents = _load_document_cache(document_cache_path)
    query = build_query(question)
    documents = _retrieve(question, query, manifest_path, cache_documents)
    visual_results = [payload for payload in (_visual_payload(document) for document in documents) if payload][:4]
    answer, confidence, model_used = _deterministic_answer(question, documents, visual_results)
    citations = _citation_list(documents)
    debug_info = {}
    if include_debug:
        debug_info = {
            "llamaindex": {
                "storage_dir": str(storage_dir),
                "document_cache_path": str(document_cache_path),
                "manifest_path": str(manifest_path),
                "vector_backend": "pinecone",
                "retrieved_count": len(documents),
                "visual_results_count": len(visual_results),
                "query_build": {
                    "route": query.route,
                    "namespace_order": query.namespace_order,
                    "figure_id": query.figure_id,
                    "visual_kind": query.visual_kind,
                    "year": query.year,
                    "metric_family": query.metric_family,
                    "query_tokens": query.query_tokens,
                },
                "available_namespaces": sorted(namespace_configs),
                "cwd": os.getcwd(),
                "repo_root": str(REPO_ROOT),
            }
        }
    log_event(
        logger,
        logging.INFO,
        "llamaindex_query_completed",
        session_id=session_id,
        question=question,
        route=query.route,
        vector_backend="pinecone",
        retrieved_count=len(documents),
        visual_results_count=len(visual_results),
        model_used=model_used,
    )
    return LlamaIndexAnswer(
        answer=answer,
        confidence_score=confidence,
        source_citations=citations,
        sources=_sources(documents),
        contexts=[document.page_content for document in documents],
        retrieved_chunks=[_chunk_payload(document) for document in documents],
        visual_results=visual_results,
        debug_info=debug_info,
        model_used=model_used,
        retrieval_mode=f"llamaindex:{query.route}",
        fallback_reasons=[],
    )
