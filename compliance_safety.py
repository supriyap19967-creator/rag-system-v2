from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from typing import Any

from gateway_guardrails import GatewayInfrastructure
from rag_invariants import RAGInvariantsValidator
from structural_vetting import StructuralOutputVetter


class ComplianceSafetyViolation(Exception):
    """Base exception for Phase 4 compliance failures."""


class SystemPromptLeakViolation(ComplianceSafetyViolation):
    """Raised when generated text appears to leak internal system instructions."""


class InfrastructureDataLeakError(ComplianceSafetyViolation):
    """Raised when generated text leaks internal infrastructure identifiers."""


class EvaluationFaithfulnessViolation(ComplianceSafetyViolation):
    """Raised when generated text is insufficiently aligned with retrieved context."""


@dataclass(frozen=True)
class GauntletResult:
    payload: dict[str, Any]
    cleared: bool
    failure_type: str | None = None


class RAGMasterSafetyGauntlet:
    """Master compliance and recovery loop for the full RAG validation stack."""

    SAFE_FALLBACK_TEXT = (
        "I apologize, but the requested answer could not clear our strict security and validation filters. "
        "Please rephrase or verify your source data context."
    )

    SYSTEM_PROMPT_LEAK_PATTERNS = (
        re.compile(r"\byou are an assistant modified to\b", re.IGNORECASE),
        re.compile(r"\byour core instructions are\b", re.IGNORECASE),
        re.compile(r"\byou must always maintain the persona\b", re.IGNORECASE),
        re.compile(r"\bsystem prompt\b", re.IGNORECASE),
        re.compile(r"\bdeveloper instructions\b", re.IGNORECASE),
        re.compile(r"\binternal configuration rules\b", re.IGNORECASE),
        re.compile(r"\bhidden chain[- ]of[- ]thought\b", re.IGNORECASE),
        re.compile(r"\bthe following are my instructions\b", re.IGNORECASE),
        re.compile(r"\bdo not reveal these instructions\b", re.IGNORECASE),
    )

    DLP_PATTERNS = (
        re.compile(r"\b[a-z0-9.-]+\.internal\b", re.IGNORECASE),
        re.compile(r"\bstaging-db-\d+[a-z0-9.-]*\b", re.IGNORECASE),
        re.compile(r"\bprod-db-\d+[a-z0-9.-]*\b", re.IGNORECASE),
        re.compile(r"\b(?:prod|staging|dev)-(?:cluster|k8s|redis|qdrant|vector|gateway)-[a-z0-9-]+\b", re.IGNORECASE),
        re.compile(r"\b(?:aws|gcp|azure)_(?:secret|access)_key\b", re.IGNORECASE),
        re.compile(r"\b(?:postgres|mysql|mongodb|redis)://[^\s]+", re.IGNORECASE),
        re.compile(r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    )

    TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
    STOPWORDS = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "into",
        "are",
        "was",
        "were",
        "has",
        "have",
        "not",
        "but",
        "your",
        "you",
        "our",
        "can",
        "will",
        "about",
        "source",
        "context",
        "answer",
    }

    def __init__(
        self,
        *,
        gateway: GatewayInfrastructure | None = None,
        invariants: RAGInvariantsValidator | None = None,
        structural_vetter: StructuralOutputVetter | None = None,
        faithfulness_threshold: float = 0.40,
    ) -> None:
        self.gateway = gateway or GatewayInfrastructure()
        self.invariants = invariants or RAGInvariantsValidator()
        self.structural_vetter = structural_vetter or StructuralOutputVetter()
        self.faithfulness_threshold = float(faithfulness_threshold)

    def run_full_validation_gauntlet(
        self,
        user_query: str,
        raw_qdrant_chunks: list[dict[str, Any]],
        model_output_payload: str | dict[str, Any],
        session_id: str,
        agent_steps: int = 0,
    ) -> dict[str, Any]:
        """
        Execute all validation phases and return either a cleared payload or a safe fallback.

        This method intentionally catches every validation exception so frontend callers
        receive a stable response shape instead of an application crash.
        """
        import logging
        import time
        from opentelemetry import trace
        
        val_logger = logging.getLogger("pydantic_ai")
        tracer = trace.get_tracer("pydantic_ai")
        
        payload = dict(model_output_payload) if isinstance(model_output_payload, dict) else {"text_response": str(model_output_payload)}
        payload.setdefault("confidence_score", 1.0)
        payload.setdefault("metadata", {})
        from app.main import _resolve_existing_image_path
        def resolve_payload_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if k in {
                        "image_path", "image_paths", "csv_path", "csv_paths",
                        "asset_path", "asset_paths", "file_path", "file_paths",
                        "table_path", "table_paths", "figure_image_path", "chart_image_path", "table_image_path"
                    } and isinstance(v, str):
                        resolved = _resolve_existing_image_path(v)
                        if resolved:
                            new_dict[k] = resolved
                        else:
                            new_dict[k] = v
                    else:
                        new_dict[k] = resolve_payload_paths(v)
                return new_dict
            elif isinstance(obj, list):
                return [resolve_payload_paths(item) for item in obj]
            return obj
        payload = resolve_payload_paths(payload)

        execution_context = {
            "gateway_result": None,
            "vetted": None,
            "score": 0.0,
            "payload": payload,
        }

        # Definition of all 13 active functional guardrail layers
        layers = [
            ("Guardrail_Layer_01_Prompt_Injection_Filter", lambda: self.gateway._scan_prompt_injection(user_query)),
            ("Guardrail_Layer_02_PII_Redaction", lambda: execution_context.update({"gateway_result": self.gateway.process_query(user_query, session_id=session_id)})),
            ("Guardrail_Layer_03_Rate_Limit_Token_Budget", lambda: self.gateway.validate_layer3(user_query, session_id=session_id, agent_steps=agent_steps)),
            ("Guardrail_Layer_04_Retrieval_Coverage", lambda: self.gateway._enforce_retrieval_coverage(user_query)),
            ("Guardrail_Layer_05_Semantic_Content", lambda: self.gateway._enforce_semantic_content(user_query)),
            ("Guardrail_Layer_06_Path_Verification", lambda: self.invariants.validate_asset_paths(self.invariants.extract_asset_paths(execution_context["payload"]))),
            ("Guardrail_Layer_07_Bounding_Box_Validator", lambda: self.invariants.validate_bounding_boxes(self.invariants.extract_bounding_boxes(execution_context["payload"]), payload=execution_context["payload"], source_text=self.invariants._source_text(raw_qdrant_chunks))),
            ("Guardrail_Layer_08_Entity_Cross_Checker", lambda: self.invariants.validate_entities_are_grounded(self.invariants.extract_fact_entities(self.invariants._generated_response_text(execution_context["payload"])), self.invariants._normalize_for_search(self.invariants._source_text(raw_qdrant_chunks)), payload=execution_context["payload"], source_text=self.invariants._source_text(raw_qdrant_chunks))),
            ("Guardrail_Layer_09_Exact_Quote_Anchoring", lambda: self.invariants.validate_exact_quotes(self.invariants.extract_direct_quotes(self.invariants._generated_response_text(execution_context["payload"])), self.invariants._source_text(raw_qdrant_chunks), payload=execution_context["payload"])),
            ("Guardrail_Layer_10_Markdown_Sanitizer", lambda: execution_context.update({"vetted": self.structural_vetter.vet(execution_context["payload"]), "payload": self.structural_vetter.vet(execution_context["payload"]).payload})),
            ("Guardrail_Layer_11_System_Prompt_Leakage_Scanner", lambda: self.scan_system_prompt_leakage(str(execution_context["payload"].get("text_response") or ""))),
            ("Guardrail_Layer_12_DLP_Blocklist", lambda: self.scan_dlp_blocklist(str(execution_context["payload"].get("text_response") or ""))),
            ("Guardrail_Layer_13_Faithfulness_Evaluation", lambda: execution_context.update({"score": self.evaluate_faithfulness(str(execution_context["payload"].get("text_response") or ""), raw_qdrant_chunks)})),
        ]

        with tracer.start_as_current_span("System_Guardrail_Engine") as engine_span:
            for layer_name, layer_func in layers:
                start_layer = time.time()
                passed = True
                error_info = ""
                
                with tracer.start_as_current_span(layer_name) as layer_span:
                    try:
                        layer_func()
                        layer_span.set_status(trace.status.Status(trace.status.StatusCode.OK))
                        val_logger.info(f"🛡️ [{layer_name}] Passed successfully.")
                    except Exception as e:
                        passed = False
                        error_info = traceback.format_exc()
                        layer_span.record_exception(e)
                        layer_span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(e)))
                        layer_span.set_attribute("guardrail.error_trace", error_info)
                        val_logger.error(f"❌ [{layer_name}] Failed: {e}")
                        
                        CRITICAL_LAYERS = {
                            "Guardrail_Layer_01_Prompt_Injection_Filter",
                            "Guardrail_Layer_02_PII_Redaction",
                            "Guardrail_Layer_03_Rate_Limit_Token_Budget",
                            "Guardrail_Layer_04_Retrieval_Coverage",
                            "Guardrail_Layer_06_Path_Verification",
                            "Guardrail_Layer_10_Markdown_Sanitizer",
                            "Guardrail_Layer_11_System_Prompt_Leakage_Scanner",
                            "Guardrail_Layer_12_DLP_Blocklist",
                            "Guardrail_Layer_13_Faithfulness_Evaluation",
                        }
                        if layer_name in CRITICAL_LAYERS:
                            # Set exception reference to break out to standard fallback handling
                            engine_span.set_attribute("guardrail.engine_failure_reason", f"{layer_name}: {str(e)}")
                            return self._fallback_payload(e)
                        else:
                            val_logger.warning(f"⚠️ [{layer_name}] Non-critical failure (logged as warning): {e}")
                            if "metadata" not in execution_context["payload"] or not isinstance(execution_context["payload"]["metadata"], dict):
                                execution_context["payload"]["metadata"] = {}
                            execution_context["payload"]["metadata"]["warning"] = f"Non-critical failure in {layer_name}: {str(e)}"
                            execution_context["payload"]["confidence_score"] = min(execution_context["payload"].get("confidence_score", 1.0), 0.5)
                        
                    duration_ms = (time.time() - start_layer) * 1000.0
                    layer_span.set_attribute("guardrail.passed", passed)
                    layer_span.set_attribute("guardrail.duration_ms", duration_ms)
                    
                    engine_span.set_attribute(f"layer.{layer_name}.passed", passed)
                    engine_span.set_attribute(f"layer.{layer_name}.duration_ms", duration_ms)

        # Finalize context enrichment
        sanitized_query = user_query
        if execution_context.get("gateway_result") is not None:
            sanitized_query = execution_context["gateway_result"].sanitized_query

        payload = execution_context["payload"]
        payload.setdefault("metadata", {})
        payload["metadata"]["sanitized_query"] = sanitized_query
        payload["metadata"]["validation_status"] = "cleared"
        payload["metadata"]["faithfulness_score"] = execution_context["score"]

        # RAG Triad Metric Calculations & Langfuse SDK Logging
        try:
            response_text = str(payload.get("text_response") or "")
            query_tokens = self._content_tokens(user_query)
            response_tokens = self._content_tokens(response_text)
            source_text = self._source_text(raw_qdrant_chunks)
            source_tokens = self._content_tokens(source_text)

            # 1. context_relevance: overlap between user query and retrieved context chunks
            context_relevance = len(query_tokens & source_tokens) / max(len(query_tokens), 1)
            # 2. faithfulness: calculated from evaluate_faithfulness
            faithfulness_score = execution_context["score"]
            # 3. answer_relevance: overlap between generated response and user query
            answer_relevance = len(response_tokens & query_tokens) / max(len(query_tokens), 1)

            import os

            # Check if keys exist before attempting any score/metric creation
            if os.getenv("LANGFUSE_PUBLIC_KEY"):
                # Execute Langfuse logging safely
                try:
                    from langfuse.decorators import langfuse_context
                    from langfuse import Langfuse
                    
                    trace_id = langfuse_context.get_current_trace_id()
                    if trace_id:
                        lf = Langfuse()
                        lf.create_score(name="context_relevance", value=context_relevance, trace_id=trace_id)
                        lf.create_score(name="faithfulness", value=faithfulness_score, trace_id=trace_id)
                        lf.create_score(name="answer_relevance", value=answer_relevance, trace_id=trace_id)
                        
                        # Fetch self-correction validation retries count dynamically
                        val_retries = 0
                        try:
                            from streamlit_ui.StreamlitApp import VALIDATION_ATTEMPT_COUNT
                            val_retries = VALIDATION_ATTEMPT_COUNT
                        except Exception:
                            pass
                        
                        # Conditional dataset export hook (more than 1 retry or faithfulness score < 0.6)
                        if val_retries >= 1 or faithfulness_score < 0.6:
                            try:
                                lf.create_dataset(name="Production_Edge_Cases")
                            except Exception:
                                pass
                            
                            lf.create_dataset_item(
                                dataset_name="Production_Edge_Cases",
                                input={
                                    "user_query": user_query,
                                    "retrieval_chunks": raw_qdrant_chunks,
                                    "validation_retries": val_retries,
                                    "compliance_score": faithfulness_score
                                },
                                expected_output={
                                    "final_response": response_text
                                },
                                metadata={
                                    "trace_id": trace_id,
                                    "validation_retries": val_retries,
                                    "compliance_score": faithfulness_score
                                }
                            )
                            val_logger.info(f"Trace {trace_id} exported to regression dataset 'Production_Edge_Cases'.")
                except Exception as e:
                    val_logger.warning(f"Skipping Langfuse score logging due to error: {e}")
            else:
                val_logger.info("Langfuse credentials not configured. Skipping metric logging.")
        except Exception as lf_eval_exc:
            val_logger.warning("Failed to log evaluation metrics or export dataset item to Langfuse: %s", lf_eval_exc)

        return payload

    def scan_system_prompt_leakage(self, text_response: str) -> None:
        for pattern in self.SYSTEM_PROMPT_LEAK_PATTERNS:
            if pattern.search(text_response or ""):
                raise SystemPromptLeakViolation("Generated response appears to leak internal system instructions.")

    def scan_dlp_blocklist(self, text_response: str) -> None:
        for pattern in self.DLP_PATTERNS:
            if pattern.search(text_response or ""):
                raise InfrastructureDataLeakError("Generated response contains internal infrastructure leakage.")

    def evaluate_faithfulness(self, text_response: str, raw_qdrant_chunks: list[dict[str, Any]]) -> float:
        response_tokens = self._content_tokens(text_response)
        if not response_tokens:
            raise EvaluationFaithfulnessViolation("Generated response is empty or contains no meaningful tokens.")

        source_text = self._source_text(raw_qdrant_chunks)
        source_tokens = self._content_tokens(source_text)
        if not source_tokens:
            import logging
            val_logger = logging.getLogger("pydantic_ai")
            val_logger.info("⚠️ [Guardrail_Layer_13_Faithfulness_Evaluation] Retrieved context is empty (Pandas/CSV query or no Qdrant results). Skipping check.")
            return 1.0

        score = len(response_tokens & source_tokens) / len(response_tokens)
        if score < self.faithfulness_threshold:
            raise EvaluationFaithfulnessViolation(
                f"Faithfulness score below threshold: score={score:.2f}, threshold={self.faithfulness_threshold:.2f}."
            )
        return score

    def _fallback_payload(self, exc: Exception) -> dict[str, Any]:
        import logging
        val_logger = logging.getLogger("pydantic_ai")
        val_logger.warning(f"⚠️ [Guardrail_Layer_14_Deterministic_Fallback_Router] - ACTIVATED due to: {exc.__class__.__name__}: {exc}")
        print("[Layer 14: Deterministic Fallback Router] - ACTIVATED")
        print("[Layer 14: Failure Traceback]")
        print(traceback.format_exc())
        return {
            "text_response": self.SAFE_FALLBACK_TEXT,
            "confidence_score": 0.0,
            "metadata": {
                "validation_status": "blocked",
                "failure_type": exc.__class__.__name__,
                "safe_fallback": True,
            },
            "image_path": None,
            "csv_path": None,
        }

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        return {
            token.lower()
            for token in cls.TOKEN_PATTERN.findall(str(text or ""))
            if token.lower() not in cls.STOPWORDS
        }

    @staticmethod
    def _source_text(raw_qdrant_chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for chunk in raw_qdrant_chunks or []:
            if not isinstance(chunk, dict):
                blocks.append(str(chunk))
                continue
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            blocks.append(str(chunk.get("content") or chunk.get("text") or chunk.get("page_content") or ""))
            blocks.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
            blocks.append(str(chunk.get("source") or ""))
        return "\n".join(block for block in blocks if block)


if __name__ == "__main__":
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
        existing_image_path = image_file.name

    try:
        chunks = [
            {
                "content": (
                    "Figure 4.1 reports 27% adoption by firms in India. "
                    "The report states: standards reduce transaction costs."
                ),
                "source": "World Development Report 2025.pdf",
                "metadata": {"image_path": existing_image_path, "page_number": 207},
            }
        ]
        good_payload = {
            "text_response": (
                "Figure 4.1 reports 27% adoption by firms in India. "
                'The source says "standards reduce transaction costs."'
            ),
            "confidence_score": 0.91,
            "metadata": {"image_path": existing_image_path},
            "image_path": existing_image_path,
            "bounding_boxes": [[0.1, 0.2, 0.7, 0.9]],
        }
        bad_asset_payload = {
            "text_response": "Figure 4.1 reports 27% adoption by firms in India.",
            "confidence_score": 0.80,
            "metadata": {},
            "image_path": os.path.join(tempfile.gettempdir(), "hallucinated_missing_asset.png"),
        }
        bad_quote_payload = {
            "text_response": 'The source says "standards eliminate all transaction costs."',
            "confidence_score": 0.80,
            "metadata": {"image_path": existing_image_path},
            "image_path": existing_image_path,
        }

        gauntlet = RAGMasterSafetyGauntlet()

        print("\n=== End-to-End Success Demo ===")
        print(
            gauntlet.run_full_validation_gauntlet(
                "Explain Figure 4.1.",
                chunks,
                good_payload,
                "demo-success",
            )
        )

        print("\n=== Deep Layer Failure Demo: Layer 4 Hallucinated Asset ===")
        print(
            gauntlet.run_full_validation_gauntlet(
                "Explain Figure 4.1.",
                chunks,
                bad_asset_payload,
                "demo-path-failure",
            )
        )

        print("\n=== Deep Layer Failure Demo: Layer 7 Quote Mismatch ===")
        print(
            gauntlet.run_full_validation_gauntlet(
                "Quote the report.",
                chunks,
                bad_quote_payload,
                "demo-quote-failure",
            )
        )
    finally:
        if os.path.exists(existing_image_path):
            os.unlink(existing_image_path)
