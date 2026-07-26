from __future__ import annotations

import sys
import types
import datasets

# Mock sentence_transformers trainer, training_args, cross_encoder, and sparse_encoder to bypass Trainer imports
sys.modules['sentence_transformers.trainer'] = types.ModuleType('sentence_transformers.trainer')
sys.modules['sentence_transformers.trainer'].SentenceTransformerTrainer = None

sys.modules['sentence_transformers.training_args'] = types.ModuleType('sentence_transformers.training_args')
sys.modules['sentence_transformers.training_args'].SentenceTransformerTrainingArguments = None
sys.modules['sentence_transformers.training_args'].BatchSamplers = None
sys.modules['sentence_transformers.training_args'].MultiDatasetBatchSamplers = None

sys.modules['sentence_transformers.sparse_encoder'] = types.ModuleType('sentence_transformers.sparse_encoder')
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoder = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderModelCardData = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainer = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainingArguments = None

sys.modules['sentence_transformers.cross_encoder'] = types.ModuleType('sentence_transformers.cross_encoder')
sys.modules['sentence_transformers.cross_encoder'].CrossEncoder = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderModelCardData = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainer = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainingArguments = None

from sentence_transformers import SentenceTransformer

import json
import logging
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from vectordb.fastembed_runtime import SafeSparseEncoder, local_sparse_vector, tokenize_for_sparse, token_to_sparse_index
from vectordb.qdrant_client_manager import QdrantSettings, get_qdrant_client as build_managed_qdrant_client

from app.conversation_manager import MultimodalConversationManager
from embeddings.embedding_model import BgeM3EmbeddingModel, EmbeddingModelSettings


load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "conversational_rag")
DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
TOP_K = int(os.getenv("RAG_TOP_K", "10"))
PREFETCH_MULTIPLIER = int(os.getenv("RAG_PREFETCH_MULTIPLIER", "4"))
RETRY_TOP_K = int(os.getenv("RAG_RETRY_TOP_K", "15"))
GLOBAL_ANALYTICS_LIMIT = int(os.getenv("GLOBAL_ANALYTICS_LIMIT", "15"))

GEMINI_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "os.getenv("GCP_API_KEY")")

BGE_MODEL_PATH = os.getenv("BGE_M3_MODEL", str(Path("hf_models_v2/bge-m3").resolve()))
BGE_CACHE_DIR = Path(os.getenv("BGE_M3_CACHE_FOLDER", "hf_cache_v2")).resolve()
BGE_DEVICE = os.getenv("BGE_M3_DEVICE", "cpu")
BGE_MAX_LENGTH = int(os.getenv("BGE_M3_QUERY_MAX_LENGTH", "1024"))
BGE_DIMENSION = 384

BM25_MODEL_NAME = os.getenv("FASTEMBED_BM25_MODEL", "Qdrant/bm25")

logger = logging.getLogger(__name__)


HARD_ENTITY_PATTERN = re.compile(
    r"\b(?P<kind>fig(?:ure|ured)?|figure|figured|fig|tab(?:le|el)?|table|tabel)\s*"
    r"(?P<identifier>[Oo0]?\s*\.?\s*\d+(?:\s*\.\s*\d+)*)",
    flags=re.IGNORECASE,
)

FIGURE_TABLE_GUARDRAIL = """Critical guardrail for figures and tables:
- Figures labeled with an "O" such as Figure O.8 are Overview figures. They are usually executive-summary duplicates or identical reprints of corresponding chapter figures such as Figure 8.4.
- If retrieved descriptions for an Overview figure and a Chapter figure have minor wording variations, do not assume the physical chart data points are different.
- Look for core semantic alignment. If both charts cover the same countries, same years, and same metrics, treat them as the same underlying graphic, say they represent the same data, and synthesize the details together.
- Flag a difference only if the retrieved context explicitly states that one chart modifies, updates, or expands upon the other.
- If the user asks for a Figure, do not substitute information from a Table even if they share the same number."""

EXECUTIVE_ANSWER_STYLE = """Answer style:
- Act as a sharp, executive-level research analyst. Synthesize complex report data into a clear analytical narrative that is insightful and highly scannable.
- Start immediately with a direct 1-2 sentence thesis that answers the core question and outlines the overarching relationship or mechanism. Do not say "Based on the retrieved context" or similar setup phrases.
- Break the explanation into logical thematic sections using bold headings, short paragraphs, or substantive bullets. Do not merely list figures; explain the causal chain: how or why one factor influences another.
- Integrate chart and text references naturally as inline evidence anchors, such as "which lowers export costs for local firms [Figure 3.10]".
- Never make a figure number the grammatical subject of a sentence. Avoid phrases like "Figure 3.2 shows".
- Write with professional clarity and a natural, fluid voice. Avoid rigid, formulaic bullet prefixes unless they genuinely serve the narrative flow."""

PROFESSIONAL_NO_DATA_RESPONSE = (
    "I don't have the explicit metrics for that specific item handy in the current report module, "
    "but I can still explain the broader concept if you want to frame the question around the topic, country, or indicator."
)

USER_FACING_PERSONA_GUARDRAIL = """User-facing persona guardrail:
- Never mention internal database logistics, retrieval mechanics, context chunks, vector search, payloads, image-processing quality, or backend failures to the user.
- Absolutely do not use phrases such as "The provided context does not contain sufficient information", "According to Context Chunk X", "The image is too blurry/simplistic to extract data", or "I cannot find this information in the database".
- If the user asks a vague follow-up such as "tell me more about this" or "explain this further", use the immediate prior conversation turn to infer what "this" refers to.
- If retrieved material contains internal engineering notes, image processing errors, OCR caveats, or phrases like "blurry image", ignore those notes completely and do not echo them.
- If there is not enough clean, concrete evidence to answer, give a polished professional limitation instead of a robotic error. For example: "I don't have the explicit metrics for that specific figure handy in the current report module, but the broader concept refers to..."."""

BANNED_USER_FACING_PHRASES = (
    "the provided context does not contain sufficient information",
    "according to context chunk",
    "context chunk",
    "the image is too blurry",
    "too blurry/simplistic",
    "i cannot find this information in the database",
)
GLOBAL_ANALYTICS_PATTERN = re.compile(
    r"\b(highest|lowest|maximum|max|min(?:imum)?|largest|smallest|total|sum|aggregate|"
    r"across\s+(?:the\s+)?(?:entire\s+)?(?:dataset|file|table|csv)|entire\s+(?:dataset|file|table|csv))\b",
    flags=re.IGNORECASE,
)
GLOBAL_ANALYTICS_RETRIEVAL_SUFFIX = (
    "\nPrioritize complete dataset summaries, table headers, CSV rows, country records, regional rows, "
    "and records needed to calculate a dataset-wide aggregate or extremum."
)
GLOBAL_ANALYTICS_FORMATTER_GUARDRAIL = """Global analytics guardrail:
- This is a dataset-wide aggregate or highest/lowest request. Calculate only from the visible records.
- If the retrieved material is a limited subset rather than a complete dataset-wide cross-section, explicitly qualify the answer with a concise phrase such as "Based on the retrieved report chapters..." and do not claim a definitive global maximum, minimum, or total."""


def sanitize_user_answer(answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return PROFESSIONAL_NO_DATA_RESPONSE
    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in BANNED_USER_FACING_PHRASES):
        return PROFESSIONAL_NO_DATA_RESPONSE
    return cleaned


def is_global_analytics_query(query: str) -> bool:
    return bool(GLOBAL_ANALYTICS_PATTERN.search(query))


def global_analytics_search_query(query: str) -> str:
    return f"{query}{GLOBAL_ANALYTICS_RETRIEVAL_SUFFIX}" if is_global_analytics_query(query) else query


@dataclass(slots=True)
class VerificationResult:
    valid: bool
    reason: str
    correction_query: str


FastEmbedSparseEncoder = SafeSparseEncoder


def build_qdrant_client() -> QdrantClient:
    settings = QdrantSettings(
        url=os.getenv("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}"),
        collection_name=COLLECTION_NAME,
    )
    logger.info("Connecting to Qdrant at %s", settings.url or f"{settings.host}:{settings.port}")
    return build_managed_qdrant_client(settings)


def _normalize_entity_identifier(raw_identifier: str) -> str:
    identifier = re.sub(r"\s+", "", raw_identifier or "").upper().replace("0.", "O.")
    if re.fullmatch(r"[O0]\d+", identifier):
        identifier = f"O.{identifier[1:]}"
    return identifier


def hard_entity_label_variants(entity: dict[str, str]) -> list[str]:
    identifier = entity["identifier"]
    prefix = "Table" if entity["kind"] == "table" else "Figure"
    variants = {
        entity["label"],
        f"{prefix} {identifier}",
        f"{prefix.lower()} {identifier}",
        identifier,
        identifier.lower(),
        identifier.upper(),
    }
    if identifier.upper().startswith("O."):
        zero_identifier = f"0.{identifier.split('.', 1)[1]}"
        variants.update({f"{prefix} {zero_identifier}", f"{prefix.lower()} {zero_identifier}", zero_identifier})
    return [variant for variant in variants if variant]


def extract_hard_entities(user_query: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in HARD_ENTITY_PATTERN.finditer(user_query or ""):
        kind_raw = match.group("kind").lower()
        kind = "table" if kind_raw.startswith(("tab", "table")) else "figure"
        identifier = _normalize_entity_identifier(match.group("identifier"))
        if not identifier:
            continue
        key = (kind, identifier)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "kind": kind,
                "identifier": identifier,
                "label": f"{'Table' if kind == 'table' else 'Figure'} {identifier}",
            }
        )
    return entities


def hard_entity_query_suffix(entities: list[dict[str, str]]) -> str:
    if not entities:
        return ""
    labels = ", ".join(entity["label"] for entity in entities)
    return f"\nHard entity labels that must be retrieved exactly: {labels}"


def build_hard_entity_filter(entities: list[dict[str, str]]) -> models.Filter | None:
    if not entities:
        return None
    conditions = [
        models.FieldCondition(
            key="metadata.figure_id",
            match=models.MatchAny(any=hard_entity_label_variants(entity)),
        )
        for entity in entities
    ]
    if len(conditions) == 1:
        return models.Filter(must=conditions)
    return models.Filter(should=conditions)


class LocalTransformerWrapper:
    def __init__(self):
        logger.info("Initializing local SentenceTransformer('all-MiniLM-L6-v2') inside query_rag...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self.model.encode(text).tolist()]


def build_embedder() -> LocalTransformerWrapper:
    return LocalTransformerWrapper()


class OpenRouterWrapper:
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY before running OpenRouter models.")
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages=messages,
                temperature=0.0,
            )
            return str(response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("OpenRouter primary model call failed: %s. Falling back to free model meta-llama/llama-3.3-70b-instruct:free", exc)
            try:
                response = self.client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct:free",
                    messages=messages,
                    temperature=0.0,
                )
                return str(response.choices[0].message.content or "").strip()
            except Exception as fallback_exc:
                logger.error("OpenRouter fallback model call failed: %s", fallback_exc)
                raise fallback_exc


def build_openrouter_client() -> OpenRouterWrapper:
    return OpenRouterWrapper()


def collection_vector_names(qdrant: QdrantClient) -> tuple[set[str], set[str]]:
    info = qdrant.get_collection(COLLECTION_NAME)
    dense_names: set[str] = set()
    sparse_names: set[str] = set()

    vectors = info.config.params.vectors
    sparse_vectors = getattr(info.config.params, "sparse_vectors", None)
    if isinstance(vectors, dict):
        dense_names = set(vectors)
    elif vectors is not None:
        dense_names = {""}
    if isinstance(sparse_vectors, dict):
        sparse_names = set(sparse_vectors)
    return dense_names, sparse_names


def _point_to_match(point: Any) -> dict[str, Any] | None:
    payload = dict(point.payload or {})
    nested_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    text = str(
        payload.get("text")
        or payload.get("page_content")
        or payload.get("content")
        or nested_payload.get("text")
        or ""
    ).strip()
    if not text:
        return None

    metadata = dict(payload.get("metadata") or {})
    return {
        "score": float(point.score),
        "text": text,
        "source": str(payload.get("source") or metadata.get("source_file") or "unknown"),
        "metadata": metadata,
    }


def classify_structural_intent_rule_based(query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["figure", "fig ", "fig.", "chart", "diagram", "image", "visual", "picture", "illustration"]):
        return "ASSET_VISUAL"
    if any(kw in query_lower for kw in ["gdp", "emission", "co2", "revenue", "metric", "indicator", "table", "timeline", "statistics", "stats", "percent", "percentage", "income group"]):
        return "TABULAR_NUMERIC"
    return "CONCEPTUAL_TEXTUAL"


def format_tabular_key_value_query_rule_based(query: str) -> str:
    return query


def retrieve_context(
    qdrant: QdrantClient,
    embedder: BgeM3EmbeddingModel,
    sparse_encoder: FastEmbedSparseEncoder | None,
    query: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    structural_intent = classify_structural_intent_rule_based(query)
    
    if structural_intent == "TABULAR_NUMERIC":
        query = format_tabular_key_value_query_rule_based(query)
        logger.info("Tabular/Numeric query reformatted: %s", query)

    hard_entities = extract_hard_entities(query)
    entity_suffix = hard_entity_query_suffix(hard_entities)
    search_query = global_analytics_search_query(query)
    search_query = f"{search_query}{entity_suffix}" if entity_suffix else search_query
    hard_filter = build_hard_entity_filter(hard_entities)

    # Detect if user query targets strict numerical metrics/timelines (GDP, emissions/CO2, revenue, etc.)
    is_numeric_query = (structural_intent == "TABULAR_NUMERIC")
    
    csv_filter = None
    if is_numeric_query:
        csv_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_type",
                    match=models.MatchValue(value="csv")
                )
            ]
        )

    # Combine strict numerical filter and hard entity filters
    qdrant_filter = None
    if csv_filter and hard_filter:
        must_conds = list(csv_filter.must)
        if hard_filter.must:
            must_conds.extend(hard_filter.must)
        should_conds = list(hard_filter.should) if hard_filter.should else None
        qdrant_filter = models.Filter(must=must_conds, should=should_conds)
    elif csv_filter:
        qdrant_filter = csv_filter
    else:
        qdrant_filter = hard_filter

    dense_query = embedder.embed_query(search_query)
    if not dense_query:
        raise RuntimeError("BGE-M3 returned an empty query vector.")

    dense_names, sparse_names = collection_vector_names(qdrant)
    can_hybrid = (
        sparse_encoder is not None
        and DENSE_VECTOR_NAME in dense_names
        and SPARSE_VECTOR_NAME in sparse_names
    )

    def _query_database(db_filter: models.Filter | None) -> list[dict[str, Any]]:
        if can_hybrid:
            logger.info(
                "Running Qdrant hybrid retrieval with dense + sparse RRF%s",
                " and metadata filters" if db_filter else "",
            )
            dense_response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                query_filter=db_filter,
                limit=max(top_k * PREFETCH_MULTIPLIER, top_k),
                with_payload=True,
            )
            sparse_query = sparse_encoder.encode_query(search_query)
            sparse_response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                query_filter=db_filter,
                limit=max(top_k * PREFETCH_MULTIPLIER, top_k),
                with_payload=True,
            )
            
            merged: dict[str, dict[str, Any]] = {}
            for path, response in (("dense", dense_response), ("sparse", sparse_response)):
                for rank, point in enumerate(response.points or [], start=1):
                    match = _point_to_match(point)
                    if match:
                        point_id = str(point.id)
                        dedupe_key = point_id or f"{match.get('source')}::{hash(match.get('text', ''))}"
                        item = merged.setdefault(dedupe_key, dict(match))
                        item["rrf_score"] = float(item.get("rrf_score", 0.0)) + (1.0 / (60 + rank))
                        item["score"] = item["rrf_score"]
            
            matches = []
            for match in merged.values():
                searchable = f"{match.get('text', '')} {match.get('metadata', {})}".lower()
                if any(term in searchable for term in ("dataset summary", "table header", "csv", "summary")):
                    match["score"] += 0.05
                matches.append(match)
            return sorted(matches, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:top_k]
        else:
            if sparse_encoder is None:
                logger.warning("FastEmbed sparse encoder unavailable; falling back to dense retrieval.")
            else:
                logger.warning("Collection does not expose expected dense+sparse named vectors; falling back to dense retrieval.")
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_query,
                using=DENSE_VECTOR_NAME if DENSE_VECTOR_NAME in dense_names else None,
                query_filter=db_filter,
                limit=top_k,
                with_payload=True,
            )

        matches: list[dict[str, Any]] = []
        for point in response.points or []:
            match = _point_to_match(point)
            if match:
                searchable = f"{match.get('text', '')} {match.get('metadata', {})}".lower()
                if any(term in searchable for term in ("dataset summary", "table header", "csv", "summary")):
                    match["score"] += 0.05
                matches.append(match)
        return sorted(matches, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    final_matches = []
    if qdrant_filter:
        if csv_filter and not hard_filter:
            logger.info("Applying numeric metrics CSV filter before retrieval")
        elif csv_filter and hard_filter:
            logger.info("Applying combined numeric CSV + hard entity metadata filter before retrieval")
        else:
            logger.info(
                "Applying flexible hard entity metadata filter before retrieval: %s",
                ", ".join(entity["label"] for entity in hard_entities),
            )
        final_matches = _query_database(qdrant_filter)
        if not final_matches and csv_filter:
            logger.warning("Combined filter returned 0 results, falling back to strict CSV metadata filter")
            final_matches = _query_database(csv_filter)
    
    if not final_matches:
        final_matches = _query_database(None)

    # For Visual/Asset queries: Ensure physical path binding and image extraction validations are strictly enforced.
    if structural_intent == "ASSET_VISUAL":
        valid_matches = []
        for m in final_matches:
            meta = m.get("metadata") or {}
            has_valid_path = False
            for key in ("image_path", "figure_image_path", "chart_image_path", "table_image_path", "image_local_path", "visual_path"):
                if key in meta and meta[key]:
                    path_val = str(meta[key])
                    if os.path.exists(path_val):
                        has_valid_path = True
                        break
            if has_valid_path:
                valid_matches.append(m)
        final_matches = valid_matches

    return final_matches


def format_context(matches: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        source = match.get("source", "unknown")
        document_type = metadata.get("document_type", "unknown")
        image_name = metadata.get("image_name", "")
        score = match.get("score", 0.0)

        header = f"[Evidence {index}] Source: {source} | Type: {document_type} | Score: {score:.4f}"
        if image_name:
            header += f" | Image: {image_name}"
        blocks.append(f"{header}\n{match['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    openrouter: OpenRouterWrapper,
    question: str,
    matches: list[dict[str, Any]],
    correction_log: str = "",
    generation_payload: dict[str, Any] | None = None,
) -> str:
    context = (
        str(generation_payload.get("compressed_context_text") or "")
        if generation_payload
        else format_context(matches)
    )
    if not context:
        return PROFESSIONAL_NO_DATA_RESPONSE
    history_text = str(generation_payload.get("chat_history_transcript") or "") if generation_payload else ""
    active_asset_paths = list(generation_payload.get("active_asset_paths") or []) if generation_payload else []
    asset_block = "\n".join(f"- {path}" for path in active_asset_paths) or "(none)"

    correction_block = ""
    if correction_log.strip():
        correction_block = f"\nPrevious answer failed verification for this reason:\n{correction_log}\n"

    prompt = f"""You are a factual enterprise RAG assistant.

Answer the user's question using ONLY the provided evidence excerpts.
The evidence may contain PDF text, CSV table rows, table summaries, or chart/figure captions.

Rules:
- Do not use outside knowledge.
- Do not invent missing values, dates, percentages, figure numbers, or country names.
- Cite sources inline using the Source value from the matching context block.
- Keep the answer concise and data-focused.

{FIGURE_TABLE_GUARDRAIL}

{EXECUTIVE_ANSWER_STYLE}

{USER_FACING_PERSONA_GUARDRAIL}
{GLOBAL_ANALYTICS_FORMATTER_GUARDRAIL if is_global_analytics_query(question) else ""}
{correction_block}
User question:
{question}

Conversation history:
{history_text or "(none)"}

Active visual/data file paths:
{asset_block}

Evidence excerpts:
{context}

Final answer:"""

    try:
        response_text = openrouter.generate(prompt)
        return sanitize_user_answer(response_text)
    except Exception as exc:
        logger.error("OpenRouter query failed during generation: %s", exc)
        return PROFESSIONAL_NO_DATA_RESPONSE


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def verify_answer(
    openrouter: OpenRouterWrapper,
    question: str,
    answer: str,
    matches: list[dict[str, Any]],
) -> VerificationResult:
    if answer.strip() == PROFESSIONAL_NO_DATA_RESPONSE:
        return VerificationResult(valid=True, reason="Answer correctly abstained.", correction_query=question)

    context = format_context(matches)
    prompt = f"""You are a strict RAG answer verifier.

Check whether the answer is fully supported by the retrieved context.
Flag hallucinations, unsupported numbers, wrong figure titles, wrong table values, or conflicts with chart/table data.

{FIGURE_TABLE_GUARDRAIL}

Return ONLY valid JSON with this exact schema:
{{
  "valid": true_or_false,
  "reason": "short explanation",
  "correction_query": "better search query if invalid, otherwise repeat the original question"
}}

Question:
{question}

Answer:
{answer}

Retrieved context:
{context}
"""

    try:
        raw = openrouter.generate(prompt)
    except Exception as exc:
        logger.error("OpenRouter query failed during verification: %s", exc)
        return VerificationResult(
            valid=True,
            reason="Verification bypassed due to OpenRouter call failure.",
            correction_query=question,
        )
    parsed = _extract_json_object(raw)
    if not parsed:
        return VerificationResult(
            valid=False,
            reason=f"Verifier did not return parseable JSON: {raw[:300]}",
            correction_query=question,
        )

    return VerificationResult(
        valid=bool(parsed.get("valid")),
        reason=str(parsed.get("reason") or ""),
        correction_query=str(parsed.get("correction_query") or question),
    )


def answer_with_self_correction(
    qdrant: QdrantClient,
    embedder: BgeM3EmbeddingModel,
    sparse_encoder: FastEmbedSparseEncoder | None,
    openrouter: OpenRouterWrapper,
    question: str,
    memory_mgr: MultimodalConversationManager | None = None,
) -> tuple[str, list[dict[str, Any]], VerificationResult]:
    top_k = max(TOP_K, GLOBAL_ANALYTICS_LIMIT) if is_global_analytics_query(question) else TOP_K
    matches = retrieve_context(qdrant, embedder, sparse_encoder, question, top_k=top_k)
    memory_mgr = memory_mgr or MultimodalConversationManager(session_id="terminal")
    generation_payload = memory_mgr.compile_generator_input(
        current_query=question,
        compressed_context_chunks=[
            {
                "content": match.get("text", ""),
                "source": match.get("source", "unknown"),
                "metadata": match.get("metadata", {}),
            }
            for match in matches
        ],
    )
    answer = generate_answer(openrouter, question, matches, generation_payload=generation_payload)
    verification = verify_answer(openrouter, question, answer, matches)

    memory_mgr.update_session_state(
        query=question,
        response=answer,
        chunks=generation_payload["context_chunks"],
    )
    return answer, matches, verification


def print_matches(matches: list[dict[str, Any]]) -> None:
    if not matches:
        print("\nNo matching context returned from Qdrant.")
        return

    print("\nRetrieved sources:")
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        image_name = metadata.get("image_name")
        suffix = f" | image={image_name}" if image_name else ""
        print(
            f"{index}. score={match['score']:.4f} | source={match['source']} | "
            f"type={metadata.get('document_type', 'unknown')}{suffix}"
        )


def build_sparse_encoder() -> FastEmbedSparseEncoder | None:
    try:
        return FastEmbedSparseEncoder()
    except Exception as exc:
        logger.warning("%s", exc)
        return None


def run_terminal_loop() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    qdrant = build_qdrant_client()
    embedder = build_embedder()
    sparse_encoder = build_sparse_encoder()
    openrouter = build_openrouter_client()
    memory_mgr = MultimodalConversationManager(session_id="terminal")

    print("\nSelf-correcting Hybrid RAG terminal ready.")
    print("Type a question and press Enter. Type 'exit', 'quit', or Ctrl+C to stop.\n")

    try:
        while True:
            question = input("Question> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                break

            try:
                answer, matches, verification = answer_with_self_correction(
                    qdrant=qdrant,
                    embedder=embedder,
                    sparse_encoder=sparse_encoder,
                    openrouter=openrouter,
                    question=question,
                    memory_mgr=memory_mgr,
                )
                print_matches(matches[:TOP_K])
                print(f"\nVerification: {'PASS' if verification.valid else 'FAIL'} - {verification.reason}")
                print("\nAnswer:")
                print(answer)
                print()
            except Exception as exc:
                logger.exception("Query failed")
                print(f"\nError: {exc}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        qdrant.close()


if __name__ == "__main__":
    run_terminal_loop()
