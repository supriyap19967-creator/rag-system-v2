import json
import logging
import os
import random
import time
from contextvars import ContextVar
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from app.utils import log_event, log_openai_cost, usage_tokens
from app.schemas import IntentCategory, QueryIntent, SourceCitation, StructuredAnswer
from app.embeddings import get_bge_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PRIMARY_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gpt-4o")
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "gpt-4o-mini")
TOGETHER_FALLBACK_MODEL = os.getenv("TOGETHER_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "1.0"))
LLM_RETRY_MAX_SECONDS = float(os.getenv("LLM_RETRY_MAX_SECONDS", "8.0"))
LLM_CALLS_PER_QUERY_LIMIT = int(os.getenv("LLM_CALLS_PER_QUERY_LIMIT", "3"))
LLM_ATTEMPTS_PER_QUERY_LIMIT = int(os.getenv("LLM_ATTEMPTS_PER_QUERY_LIMIT", os.getenv("LLM_CALLS_PER_QUERY_LIMIT", "3")))

openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0) if OPENAI_API_KEY else None
together_client = (
    OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1", max_retries=0)
    if TOGETHER_API_KEY
    else None
)
groq_client = (
    OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", max_retries=0)
    if GROQ_API_KEY
    else None
)
_llm_disabled_reason: Optional[str] = None
_llm_attempt_count: ContextVar[int] = ContextVar("llm_attempt_count", default=0)
_llm_attempt_limit: ContextVar[int] = ContextVar("llm_attempt_limit", default=LLM_ATTEMPTS_PER_QUERY_LIMIT)
_openai_circuit_open: ContextVar[bool] = ContextVar("openai_circuit_open", default=False)
_openai_circuit_reason: ContextVar[str] = ContextVar("openai_circuit_reason", default="")

SYSTEM_PROMPT = '''You are a Financial Data Expert.
Your answers must be 100% grounded in the provided context.
If the context does not contain the answer, say: "I do not have sufficient data to answer this question."'''

INSUFFICIENT_DATA_MESSAGE = "I do not have sufficient data to answer this question."
CONTEXTUAL_SYSTEM_PROMPT = (
    "You are a conversational financial analyst. Answer strictly from the provided evidence. "
    "Use the conversation history only to preserve continuity, understand what the user already knows, "
    "and avoid repeating prior definitions or background unless the new question truly requires it. "
    "If the user asks a follow-up, build on the previous answer naturally while staying grounded in the evidence."
)


class LLMCallBudgetExceeded(RuntimeError):
    pass


class LLMRateLimitExceeded(RuntimeError):
    pass


class LLMQuotaExceeded(RuntimeError):
    pass


class LLMCircuitOpen(RuntimeError):
    pass


def reset_llm_call_counter(limit: Optional[int] = None):
    return (
        _llm_attempt_count.set(0),
        _llm_attempt_limit.set(limit or LLM_ATTEMPTS_PER_QUERY_LIMIT),
        _openai_circuit_open.set(False),
        _openai_circuit_reason.set(""),
    )


def restore_llm_call_counter(token) -> None:
    count_token, limit_token, circuit_token, reason_token = token
    _llm_attempt_count.reset(count_token)
    _llm_attempt_limit.reset(limit_token)
    _openai_circuit_open.reset(circuit_token)
    _openai_circuit_reason.reset(reason_token)


def get_llm_call_count() -> int:
    return _llm_attempt_count.get()


def _increment_llm_attempt_count(call_type: str, model: str, provider: str, session_id: Optional[str], attempt_no: int, max_attempts: int) -> int:
    next_count = _llm_attempt_count.get() + 1
    limit = _llm_attempt_limit.get()
    if next_count > limit:
        log_event(
            logger,
            logging.WARNING,
            "llm_attempt_budget_exceeded",
            session_id=session_id,
            logical_stage=call_type,
            provider=provider,
            model=model,
            attempted_attempt_count=next_count,
            limit=limit,
            attempt_no=attempt_no,
            max_attempts=max_attempts,
        )
        raise LLMCallBudgetExceeded(f"LLM outbound attempt budget exceeded for this query ({limit}).")
    _llm_attempt_count.set(next_count)
    log_event(
        logger,
        logging.INFO,
        "openai_attempt_started" if provider == "openai" else "llm_attempt_started",
        session_id=session_id,
        logical_stage=call_type,
        provider=provider,
        model=model,
        attempt_no=attempt_no,
        max_attempts=max_attempts,
        total_attempt_count=next_count,
        limit=limit,
    )
    return next_count


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw_retry_after = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    try:
        return float(raw_retry_after)
    except (TypeError, ValueError):
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def _status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return int(status_code)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return int(response_status) if response_status is not None else None


def _error_code(exc: Exception) -> str:
    code = str(getattr(exc, "code", "") or "").strip()
    if code:
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        return str(error.get("code") or error.get("type") or "").strip()
    response = getattr(exc, "response", None)
    try:
        payload = response.json() if response is not None else {}
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        error = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        return str(error.get("code") or error.get("type") or "").strip()
    return ""


def _is_quota_error(exc: Exception) -> bool:
    error_code = _error_code(exc).lower()
    error_text = str(exc).lower()
    return any(
        marker in error_code or marker in error_text
        for marker in ("insufficient_quota", "quota_exceeded")
    )


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    status_code = _status_code(exc)
    return status_code is not None and 500 <= status_code <= 599


def _provider_label(client: OpenAI) -> str:
    if client is openai_client:
        return "openai"
    if client is together_client:
        return "together"
    if client is groq_client:
        return "groq"
    return "unknown"


def _open_openai_circuit(reason: str, session_id: Optional[str], call_type: str, model: str) -> None:
    _openai_circuit_open.set(True)
    _openai_circuit_reason.set(reason)
    log_event(
        logger,
        logging.WARNING,
        "openai_circuit_breaker_opened",
        session_id=session_id,
        logical_stage=call_type,
        provider="openai",
        model=model,
        reason=reason,
        remaining_stages_skipped=True,
    )


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _source_label(metadata: Dict[str, object]) -> str:
    source_type = _safe_str(metadata.get("source_type")).lower()
    if source_type == "csv":
        return _safe_str(metadata.get("source_files")) or "csv"
    return _safe_str(metadata.get("source")) or "unknown"


def _section_label(metadata: Dict[str, object]) -> str:
    for key in ("header_path", "section_header", "h3", "h2", "h1"):
        value = _safe_str(metadata.get(key))
        if value:
            return value
    if metadata.get("year"):
        return f"year={_safe_str(metadata.get('year'))}"
    return "section=N/A"


def _metadata_summary(metadata: Dict[str, object]) -> str:
    fields: List[str] = []
    for key in (
        "source_type",
        "dataset_type",
        "country_name",
        "country_iso3",
        "country_codes",
        "indicator",
        "year",
        "value",
        "page",
        "section_index",
        "chunk_index",
    ):
        value = _safe_str(metadata.get(key))
        if value:
            fields.append(f"{key}={value}")
    return ", ".join(fields) if fields else "metadata=N/A"


def format_source_reference(index: int, document: Document) -> str:
    metadata = dict(document.metadata)
    return f"[{index}] {_source_label(metadata)} / {_section_label(metadata)} / {_metadata_summary(metadata)}"


def build_structured_citations(documents: Sequence[Document]) -> List[SourceCitation]:
    citations: List[SourceCitation] = []
    for document in documents[:5]:
        metadata = dict(document.metadata)
        raw_page = metadata.get("page")
        page_number: Optional[int]
        try:
            page_number = int(raw_page) if raw_page not in (None, "", "N/A") else None
        except (TypeError, ValueError):
            page_number = None

        citations.append(
            SourceCitation(
                filename=_source_label(metadata),
                page_number=page_number,
            )
        )
    return citations


def sanitize_citations(
    citations: Sequence[SourceCitation],
    documents: Sequence[Document],
) -> List[SourceCitation]:
    allowed = {
        (citation.filename, citation.page_number)
        for citation in build_structured_citations(documents)
    }
    sanitized: List[SourceCitation] = []
    seen = set()
    for citation in citations:
        key = (citation.filename, citation.page_number)
        if key not in allowed or key in seen:
            continue
        seen.add(key)
        sanitized.append(citation)
    return sanitized if sanitized else build_structured_citations(documents)


def build_context_blocks(documents: Sequence[Document]) -> Tuple[str, List[str]]:
    context_blocks: List[str] = []
    source_references: List[str] = []

    for index, document in enumerate(documents[:5], start=1):
        metadata = dict(document.metadata)
        source_type = _safe_str(metadata.get("source_type")).lower() or "unknown"
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"source_type: {source_type}",
                    f"filename: {_source_label(metadata)}",
                    f"section: {_section_label(metadata)}",
                    f"metadata: {_metadata_summary(metadata)}",
                    "content:",
                    document.page_content.strip(),
                ]
            )
        )
        source_references.append(format_source_reference(index, document))

    return "\n\n".join(context_blocks), source_references


def build_chat_history_block(chat_history: Sequence[Dict[str, str]]) -> str:
    if not chat_history:
        return "No prior conversation."

    lines: List[str] = []
    for message in chat_history:
        role = _safe_str(message.get("role")) or "unknown"
        content = _safe_str(message.get("content")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def build_answer_user_prompt(
    question: str,
    documents: Sequence[Document],
    chat_history: Sequence[Dict[str, str]],
    intent: Optional[IntentCategory] = None,
) -> Tuple[str, List[str]]:
    context, source_references = build_context_blocks(documents)
    conversation_history = build_chat_history_block(chat_history)
    schema_json = json.dumps(StructuredAnswer.model_json_schema(), indent=2)

    intent_directive = ""
    if intent == IntentCategory.NUMERICAL:
        intent_directive = "- If the intent is NUMERICAL, prioritize data from the CSV source. Do not ignore numerical values in favor of general text description."

    user_prompt = f"""Return only a JSON object that matches the provided schema.

Rules:
- Use only the retrieved context below.
- Use the conversation history only to preserve continuity and tone. Do not use it as factual evidence.
- Use CSV chunks for exact numerical values.
- Use PDF chunks for qualitative explanation or reasoning.
- If both CSV and PDF sources are relevant, explicitly connect the exact number from CSV with the explanation from PDF.
{intent_directive}
- The "answer" field must contain a direct answer followed by a short explanation grounded in the context.
- The "confidence_score" field must be a float between 0 and 1 based only on the completeness and consistency of the retrieved context.
- The "source_citations" field must list only filenames and page numbers from the retrieved context.
- Do not include markdown, prose outside JSON, or citation markers like [1] in the output.
- If the context is insufficient, set "answer" to "{INSUFFICIENT_DATA_MESSAGE}", set a low confidence score, and keep citations limited to the retrieved evidence.

JSON schema:
{schema_json}

Question:
{question}

Conversation history:
{conversation_history}

Retrieved context:
{context}
"""

    return user_prompt, source_references


class HybridLLM:
    def is_available(self) -> bool:
        return _llm_disabled_reason is None and any(
            client is not None for client in (openai_client, together_client, groq_client)
        )

    def classify_intent(self, query: str, session_id: Optional[str] = None) -> QueryIntent:
        if openai_client is None:
            return QueryIntent(intent=IntentCategory.HYBRID, reasoning="Default due to no client")

        prompt = f"""Classify the user's query into one of three categories:
- NUMERICAL: Questions about GDP, growth, CO2 values, or specific years.
- QUALITATIVE: Questions about laws, regulations, standards, or 'nation building'.
- HYBRID: Questions that need both (e.g., 'How do economic standards affect GDP?').

Return only a JSON object matching this schema:
{{
  "intent": "NUMERICAL | QUALITATIVE | HYBRID",
  "reasoning": "Brief explanation"
}}

Query: {query}
"""
        try:
            response = self._invoke_with_client(
                client=openai_client,
                model=QUERY_REWRITE_MODEL,
                user_prompt=prompt,
                system_prompt="You are an intent classifier for financial queries.",
                session_id=session_id,
                call_type="intent_classification",
            )
            data = json.loads(response["content"])
            return QueryIntent.model_validate(data)
        except Exception as e:
            print(f"Intent classification failed: {e}. Defaulting to HYBRID.")
            return QueryIntent(intent=IntentCategory.HYBRID, reasoning=f"Error: {e}")
    def _invoke_with_client(
        self,
        client: OpenAI,
        model: str,
        user_prompt: str,
        system_prompt: str,
        session_id: Optional[str] = None,
        call_type: str = "generation",
    ) -> Dict[str, object]:
        provider = _provider_label(client)
        if provider == "openai" and _openai_circuit_open.get():
            reason = _openai_circuit_reason.get() or "openai_unavailable_for_request"
            log_event(
                logger,
                logging.WARNING,
                "openai_stage_skipped_circuit_open",
                session_id=session_id,
                logical_stage=call_type,
                provider=provider,
                model=model,
                reason=reason,
            )
            raise LLMCircuitOpen(reason)

        max_attempts = LLM_MAX_RETRIES + 1
        last_transient_error: Optional[Exception] = None
        for attempt_index in range(max_attempts):
            attempt_no = attempt_index + 1
            _increment_llm_attempt_count(
                call_type,
                model,
                provider,
                session_id,
                attempt_no=attempt_no,
                max_attempts=max_attempts,
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "openai_attempt_finished" if provider == "openai" else "llm_attempt_finished",
                    session_id=session_id,
                    logical_stage=call_type,
                    provider=provider,
                    model=model,
                    attempt_no=attempt_no,
                    max_attempts=max_attempts,
                    status_code=200,
                    retry=False,
                    fallback_triggered=False,
                )
                break
            except Exception as exc:
                status_code = _status_code(exc)
                quota_error = _is_quota_error(exc)
                rate_limit_error = _is_rate_limit_error(exc)
                if provider == "openai" and (rate_limit_error or quota_error):
                    reason = _error_code(exc) or ("rate_limit" if rate_limit_error else "quota")
                    _open_openai_circuit(reason, session_id, call_type, model)
                    log_event(
                        logger,
                        logging.WARNING,
                        "openai_attempt_finished",
                        session_id=session_id,
                        logical_stage=call_type,
                        provider=provider,
                        model=model,
                        attempt_no=attempt_no,
                        max_attempts=max_attempts,
                        status_code=status_code,
                        error_code=_error_code(exc),
                        retry=False,
                        fallback_triggered=True,
                    )
                    if quota_error:
                        raise LLMQuotaExceeded(reason) from exc
                    raise LLMRateLimitExceeded(reason) from exc

                if not _is_transient_error(exc):
                    log_event(
                        logger,
                        logging.WARNING,
                        "openai_attempt_finished" if provider == "openai" else "llm_attempt_finished",
                        session_id=session_id,
                        logical_stage=call_type,
                        provider=provider,
                        model=model,
                        attempt_no=attempt_no,
                        max_attempts=max_attempts,
                        status_code=status_code,
                        error_code=_error_code(exc),
                        retry=False,
                        fallback_triggered=True,
                    )
                    raise

                last_transient_error = exc
                retry_after = _retry_after_seconds(exc)
                if attempt_index >= LLM_MAX_RETRIES:
                    log_event(
                        logger,
                        logging.WARNING,
                        "llm_transient_retries_exhausted",
                        session_id=session_id,
                        logical_stage=call_type,
                        provider=provider,
                        model=model,
                        attempts=attempt_no,
                        status_code=status_code,
                    )
                    raise

                backoff = retry_after if retry_after is not None else min(
                    LLM_RETRY_MAX_SECONDS,
                    LLM_RETRY_BASE_SECONDS * (2**attempt_index) + random.uniform(0, 0.35),
                )
                log_event(
                    logger,
                    logging.WARNING,
                    "openai_attempt_finished" if provider == "openai" else "llm_attempt_finished",
                    session_id=session_id,
                    logical_stage=call_type,
                    provider=provider,
                    model=model,
                    attempt_no=attempt_no,
                    max_attempts=max_attempts,
                    status_code=status_code,
                    error_code=_error_code(exc),
                    retry=True,
                    fallback_triggered=False,
                    sleep_seconds=round(backoff, 3),
                    retry_after_header=retry_after,
                )
                time.sleep(backoff)
        else:  # pragma: no cover - defensive guard for static analyzers
            raise RuntimeError("LLM call failed.") from last_transient_error

        usage = getattr(response, "usage", None)
        if session_id:
            log_openai_cost(
                session_id=session_id,
                model=model,
                call_type=call_type,
                input_tokens=usage_tokens(usage, "prompt_tokens"),
                output_tokens=usage_tokens(usage, "completion_tokens"),
            )
        return {
            "content": response.choices[0].message.content or "",
            "usage": usage,
        }

    def _fallback_target(self) -> Tuple[OpenAI, str, str]:
        if together_client is not None:
            return together_client, TOGETHER_FALLBACK_MODEL, "llama-3.3-70b-together"
        if groq_client is not None:
            return groq_client, GROQ_FALLBACK_MODEL, "llama-3.3-70b-groq"
        raise RuntimeError("Fallback model is not configured. Set TOGETHER_API_KEY or GROQ_API_KEY.")

    def embed_text(self, text: str, session_id: Optional[str] = None, call_type: str = "embedding") -> List[float]:
        return list(get_bge_embeddings().embed_query(text))

    def invoke(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        global _llm_disabled_reason

        if _llm_disabled_reason is not None:
            raise RuntimeError(_llm_disabled_reason)
        if openai_client is None:
            raise RuntimeError("Primary model is not configured. Set OPENAI_API_KEY.")

        try:
            result = self._invoke_with_client(
                client=openai_client,
                model=PRIMARY_MODEL,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                session_id=session_id,
                call_type="generation",
            )
            return {
                "answer": result["content"],
                "model_used": PRIMARY_MODEL,
            }
        except (LLMRateLimitExceeded, LLMQuotaExceeded, LLMCircuitOpen, LLMCallBudgetExceeded) as exc:
            log_event(
                logger,
                logging.WARNING,
                "openai_generation_fallback_triggered",
                session_id=session_id,
                reason=str(exc),
                fallback="local_deterministic",
            )
            return {
                "answer": f"LLM unavailable: {str(exc)}",
                "model_used": "llm-unavailable",
            }
        except Exception:
            print("GPT-4o failed. Falling back to Llama-3.3 for reliability.")
            fallback_label = "llm-unavailable"
            try:
                fallback_client, fallback_model, fallback_label = self._fallback_target()
                result = self._invoke_with_client(
                    client=fallback_client,
                    model=fallback_model,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    session_id=None,
                    call_type="generation_fallback",
                )
                return {
                    "answer": result["content"],
                    "model_used": fallback_label,
                }
            except Exception as e:
                _llm_disabled_reason = str(e)
                log_event(
                    logger,
                    logging.WARNING,
                    "llm_provider_disabled",
                    reason=_llm_disabled_reason,
                )
                return {
                    "answer": f"LLM error: {str(e)}",
                    "model_used": fallback_label,
                }

    def rewrite_query(
        self,
        user_input: str,
        chat_history: Sequence[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> str:
        if openai_client is None:
            return user_input

        conversation_history = build_chat_history_block(chat_history)
        rewrite_prompt = f"""Rewrite the latest user question into a standalone search query for retrieval.

Rules:
- Preserve the user's original intent.
- Resolve ambiguous references using the conversation history.
- Keep financial entities, country names, years, and metrics explicit when available.
- Return only the rewritten standalone query.
- If the latest question is already standalone, return it unchanged.

Conversation history:
{conversation_history}

Latest user question:
{user_input}
"""

        try:
            rewritten_query = self._invoke_with_client(
                client=openai_client,
                model=QUERY_REWRITE_MODEL,
                user_prompt=rewrite_prompt,
                system_prompt="You rewrite follow-up questions into standalone retrieval queries.",
                session_id=session_id,
                call_type="query_rewrite",
            )["content"].strip()
            return rewritten_query or user_input
        except Exception:
            return user_input

    def generate_grounded_answer(
        self,
        *,
        question: str,
        evidence_blocks: Sequence[str],
        citations: Sequence[SourceCitation],
        answer_style: str,
        chat_history: Sequence[Dict[str, str]] = (),
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        if not self.is_available():
            raise RuntimeError("No configured LLM is available for grounded answer generation.")

        schema_json = json.dumps(StructuredAnswer.model_json_schema(), indent=2)
        conversation_history = build_chat_history_block(chat_history)
        evidence_text = "\n\n".join(
            f"[Evidence {index}]\n{block.strip()}"
            for index, block in enumerate(evidence_blocks, start=1)
            if str(block or "").strip()
        )
        allowed_citations = json.dumps(
            [citation.model_dump() for citation in citations],
            indent=2,
        )
        prompt = self.get_talkative_answer(
            query=question,
            context=evidence_text or "No evidence provided.",
            history=conversation_history,
            instruction=(
                "Return only a JSON object that matches the schema below.\n\n"
                "Rules:\n"
                f"- Use only citations from this allowed list:\n{allowed_citations}\n"
                f"- JSON schema:\n{schema_json}\n"
                f"- If the evidence is insufficient, set \"answer\" to \"{INSUFFICIENT_DATA_MESSAGE}\" and use a low confidence score.\n"
                "- Do not include markdown or prose outside the JSON object.\n"
                f"- Keep the answer style aligned to this instruction: {answer_style}"
            ),
            session_id=session_id,
        )
        result = {
            "answer": prompt,
            "model_used": "unknown",
        }
        if isinstance(prompt, dict):
            result = prompt
        event_level = logging.INFO
        event_name = "grounded_llm_generation_completed"
        if str(result.get("model_used")) == "llm-unavailable":
            event_level = logging.WARNING
            event_name = "grounded_llm_generation_unavailable"
        log_event(
            logger,
            event_level,
            event_name,
            model_used=result.get("model_used"),
            evidence_block_count=len([block for block in evidence_blocks if str(block or "").strip()]),
        )
        return result

    def get_talkative_answer(
        self,
        *,
        query: str,
        context: str,
        history: str,
        instruction: str = "",
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        system_prompt = f"""
You are a helpful Financial Analyst.

PREVIOUS CONVERSATION:
{history}

NEW RESEARCH DATA:
{context}

USER'S NEW QUESTION: {query}

INSTRUCTION:
1. If the user asks a follow-up, DO NOT repeat what you said in the PREVIOUS CONVERSATION.
2. Focus ONLY on the NEW RESEARCH DATA to answer the follow-up.
3. If the first answer was a definition, and this question is about "how to", provide a strategy-oriented answer using only the new research data.
4. Keep it conversational but fully grounded.
5. Preserve exact numeric values exactly when they appear in the new research data.
6. Do not hallucinate facts, sources, years, or policies.

{instruction}
""".strip()
        return self.invoke(
            user_prompt=system_prompt,
            system_prompt=CONTEXTUAL_SYSTEM_PROMPT,
            session_id=session_id,
        )


def get_hybrid_llm() -> HybridLLM:
    return HybridLLM()


def parse_structured_answer(answer: str, documents: Sequence[Document]) -> StructuredAnswer:
    fallback = StructuredAnswer(
        answer=INSUFFICIENT_DATA_MESSAGE if not answer or len(answer.strip()) < 5 else answer.strip(),
        confidence_score=0.1,
        source_citations=build_structured_citations(documents),
    )
    if not answer:
        return fallback

    try:
        parsed = StructuredAnswer.model_validate_json(answer)
        parsed.source_citations = sanitize_citations(parsed.source_citations, documents)
        return parsed
    except Exception:
        try:
            parsed = StructuredAnswer.model_validate(json.loads(answer))
            parsed.source_citations = sanitize_citations(parsed.source_citations, documents)
            return parsed
        except Exception:
            return fallback


def validate_answer(answer: str, documents: Sequence[Document]) -> StructuredAnswer:
    return parse_structured_answer(answer, documents)
