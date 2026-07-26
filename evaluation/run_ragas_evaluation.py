import json
import math
import os
import sys
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.ragas_eval_set import EVALUATION_CASES

DEFAULT_RAGAS_BASE_URL = "http://localhost:11434/v1"
DEFAULT_RAGAS_MODEL = "llama3.1"
DEFAULT_RAGAS_API_KEY = "ollama"
DEFAULT_RAGAS_TIMEOUT_SECONDS = 300
DEFAULT_RAGAS_MAX_WORKERS = 2


def _truncate(text: str, limit: int = 90) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _print_table(rows: List[Dict[str, Any]], headers: List[str]) -> None:
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) if rows else len(header)
        for header in headers
    }
    header_row = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)
    print(header_row)
    print(divider)
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _run_live_cases() -> List[Dict[str, Any]]:
    from app import main as rag_app

    rag_app.load_models()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(EVALUATION_CASES, start=1):
        response = rag_app.query_rag(
            rag_app.QueryRequest(
                session_id=f"ragas-eval-{index}",
                question=case["query"],
            )
        )
        rows.append(
            {
                "question": case["query"],
                "category": case["category"],
                "ground_truth": case["ground_truth"],
                "answer": response.get("answer", ""),
                "contexts": response.get("contexts", []),
                "retrieval_mode": response.get("retrieval_mode", ""),
            }
        )
    return rows


def _print_live_summary(rows: List[Dict[str, Any]]) -> None:
    summary_rows = [
        {
            "category": row["category"],
            "retrieval_mode": row["retrieval_mode"],
            "contexts": len(row["contexts"]),
            "answer_preview": _truncate(row["answer"]),
        }
        for row in rows
    ]
    print("\nLive evaluation cases")
    _print_table(summary_rows, ["category", "retrieval_mode", "contexts", "answer_preview"])


def _validate_eval_rows(rows: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    diagnostics: List[Dict[str, Any]] = []
    valid = True
    for row in rows:
        contexts = row.get("contexts")
        row_valid = (
            isinstance(row.get("question"), str)
            and bool(str(row.get("question")).strip())
            and isinstance(row.get("answer"), str)
            and bool(str(row.get("answer")).strip())
            and isinstance(contexts, list)
            and all(isinstance(context, str) and bool(context.strip()) for context in contexts)
        )
        diagnostics.append(
            {
                "category": row.get("category", ""),
                "question_ok": isinstance(row.get("question"), str) and bool(str(row.get("question")).strip()),
                "answer_ok": isinstance(row.get("answer"), str) and bool(str(row.get("answer")).strip()),
                "contexts_ok": isinstance(contexts, list) and all(
                    isinstance(context, str) and bool(context.strip()) for context in (contexts or [])
                ),
                "context_count": len(contexts or []),
            }
        )
        valid = valid and row_valid
    return valid, diagnostics


def _ragas_config() -> Dict[str, str]:
    return {
        "base_url": os.getenv("RAGAS_BASE_URL", DEFAULT_RAGAS_BASE_URL).rstrip("/"),
        "model": os.getenv("RAGAS_MODEL", DEFAULT_RAGAS_MODEL),
        "api_key": os.getenv("RAGAS_API_KEY", DEFAULT_RAGAS_API_KEY),
    }


def _fetch_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_json_object(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    text = str(raw_text or "").strip()
    if not text:
        return None, "empty_output"

    if text.startswith("{") and text.endswith("}"):
        return text, None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "json_object_not_found"

    cleaned = text[start : end + 1].strip()
    if not cleaned:
        return None, "cleaned_json_empty"
    return cleaned, None


def _ollama_model_available(requested_model: str, available_models: List[str]) -> bool:
    if requested_model in available_models:
        return True
    normalized_requested = requested_model.split(":", 1)[0]
    return any(
        model == normalized_requested or model.split(":", 1)[0] == normalized_requested
        for model in available_models
    )


def _check_ollama_ready(config: Dict[str, str]) -> bool:
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    models_url = urljoin(f"{config['base_url']}/", "models")
    try:
        payload = _fetch_json(models_url, headers=headers)
    except URLError as exc:
        print("\nRAGAS evaluation could not reach Ollama.")
        print(f"Reason: {exc}")
        print("Start Ollama first, for example:")
        print("  ollama serve")
        return False
    except Exception as exc:
        print("\nRAGAS evaluation could not query the Ollama model list.")
        print(f"Reason: {exc}")
        return False

    available_models = [
        str(model.get("id") or "").strip()
        for model in payload.get("data", [])
        if str(model.get("id") or "").strip()
    ]
    if not available_models:
        print("\nOllama is reachable, but no models were reported by the OpenAI-compatible endpoint.")
        print(f"Checked endpoint: {models_url}")
        return False

    if not _ollama_model_available(config["model"], available_models):
        print("\nThe configured RAGAS judge model is not available in Ollama.")
        print(f"Requested model: {config['model']}")
        print(f"Available models: {', '.join(available_models)}")
        print(f"Pull it first if needed: ollama pull {config['model']}")
        return False

    print("\nRAGAS evaluator configuration")
    print(f"  Base URL: {config['base_url']}")
    print(f"  Model: {config['model']}")
    print("  Judge provider: Ollama (OpenAI-compatible)")
    return True


def _debug_faithfulness_probe(base_llm: object, rows: List[Dict[str, Any]]) -> None:
    from ragas.metrics._faithfulness import StatementGeneratorInput, StatementGeneratorOutput, StatementGeneratorPrompt

    probe_row = next(
        (
            row for row in rows
            if isinstance(row.get("answer"), str)
            and row["answer"].strip()
            and isinstance(row.get("contexts"), list)
            and row["contexts"]
        ),
        None,
    )
    if probe_row is None:
        print("\nFaithfulness probe skipped: no valid row with answer and contexts.")
        return

    prompt = StatementGeneratorPrompt()
    input_data = StatementGeneratorInput(
        question=str(probe_row["question"]),
        answer=str(probe_row["answer"]),
    )
    raw_response = base_llm.invoke(prompt.to_string(input_data)).content
    print("\nFaithfulness debug probe")
    print(f"  Probe category: {probe_row.get('category', 'unknown')}")
    print(f"  Raw statement-generator output preview: {_truncate(str(raw_response), 220)}")
    try:
        parsed = StatementGeneratorOutput.model_validate_json(str(raw_response))
        print(f"  Direct JSON parse: ok ({len(parsed.statements)} statements)")
    except Exception as exc:
        print(f"  Direct JSON parse failed: {exc}")
        print("  This usually means the local judge returned extra prose around JSON, which breaks the strict faithfulness parser path.")


def _ragas_runtime_config() -> Dict[str, int]:
    return {
        "timeout": int(os.getenv("RAGAS_TIMEOUT_SECONDS", str(DEFAULT_RAGAS_TIMEOUT_SECONDS))),
        "max_workers": int(os.getenv("RAGAS_MAX_WORKERS", str(DEFAULT_RAGAS_MAX_WORKERS))),
    }


def _print_faithfulness_debug(metric: "LocalOllamaFaithfulness") -> None:
    print("\nFaithfulness debug status")
    print(f"  Retry used: {metric.retry_used}")
    print(f"  Last failure reason: {metric.last_failure_reason or 'none'}")
    if metric.last_raw_output:
        print(f"  Raw output preview: {_truncate(metric.last_raw_output, 220)}")
    if metric.last_cleaned_output:
        print(f"  Cleaned output preview: {_truncate(metric.last_cleaned_output, 220)}")


def _print_runtime_config(runtime_config: Dict[str, int]) -> None:
    print("  Timeout (s): {}".format(runtime_config["timeout"]))
    print("  Max workers: {}".format(runtime_config["max_workers"]))


class LocalOllamaFaithfulnessMixin:
    last_failure_reason: Optional[str] = None
    last_raw_output: str = ""
    last_cleaned_output: str = ""
    retry_used: bool = False
    debug_events: List[Dict[str, str]] = []

    def _record_debug(self, stage: str, raw_output: str, cleaned_output: str, reason: Optional[str]) -> None:
        if not hasattr(self, "debug_events") or self.debug_events is None:
            self.debug_events = []
        self.last_raw_output = raw_output
        self.last_cleaned_output = cleaned_output
        self.last_failure_reason = reason
        self.debug_events.append(
            {
                "stage": stage,
                "reason": reason or "",
                "raw_preview": _truncate(raw_output, 180),
                "cleaned_preview": _truncate(cleaned_output, 180),
            }
        )

    async def _generate_statements_with_retry(self, row: Dict[str, Any], callbacks: Any) -> Any:
        from langchain_core.prompt_values import StringPromptValue
        from ragas.metrics._faithfulness import StatementGeneratorInput, StatementGeneratorOutput

        assert self.llm is not None, "LLM is not set"

        prompt_input = StatementGeneratorInput(question=row["user_input"], answer=row["response"])
        base_prompt = self.statement_generator_prompt.to_string(prompt_input)
        strict_suffix = (
            "\nCRITICAL OUTPUT RULES:\n"
            "- Return ONLY a valid JSON object.\n"
            "- Do not add any explanation, prefix, suffix, markdown, or notes.\n"
            "- Any text outside the JSON object is invalid.\n"
        )
        attempts = [
            ("primary", base_prompt + strict_suffix),
            (
                "retry_strict_json",
                base_prompt
                + strict_suffix
                + "\nRETRY: Your previous output was invalid. Return exactly one JSON object and nothing else.\n",
            ),
        ]

        parse_errors: List[str] = []
        self.retry_used = False
        for index, (stage, prompt_text) in enumerate(attempts):
            if index > 0:
                self.retry_used = True
            try:
                result = await self.llm.generate(
                    prompt=StringPromptValue(text=prompt_text),
                    n=1,
                    temperature=0.0,
                    callbacks=callbacks,
                )
            except Exception as exc:
                reason = f"timeout_or_generation_error:{exc.__class__.__name__}"
                self._record_debug(stage, "", "", reason)
                parse_errors.append(reason)
                continue

            raw_output = str(result.generations[0][0].text or "").strip()
            cleaned_output, extraction_error = _extract_json_object(raw_output)
            self._record_debug(stage, raw_output, cleaned_output or "", extraction_error)
            if extraction_error is not None or not cleaned_output:
                parse_errors.append(extraction_error or "unknown_json_extraction_error")
                continue

            try:
                parsed = StatementGeneratorOutput.model_validate_json(cleaned_output)
                if not parsed.statements:
                    reason = "empty_statement_list"
                    self._record_debug(stage, raw_output, cleaned_output, reason)
                    parse_errors.append(reason)
                    continue
                self.last_failure_reason = None
                return parsed
            except Exception as exc:
                reason = f"json_parse_error:{exc}"
                self._record_debug(stage, raw_output, cleaned_output, reason)
                parse_errors.append(reason)

        self.last_failure_reason = "; ".join(parse_errors) if parse_errors else "unknown_faithfulness_failure"
        raise ValueError(self.last_failure_reason)


def main() -> None:
    rows = _run_live_cases()
    _print_live_summary(rows)
    rows_valid, row_diagnostics = _validate_eval_rows(rows)
    print("\nEvaluation row validation")
    _print_table(row_diagnostics, ["category", "question_ok", "answer_ok", "contexts_ok", "context_count"])
    if not rows_valid:
        print("\nRAGAS evaluation cannot proceed because one or more rows have invalid question/answer/context inputs.")
        return

    try:
        from dataclasses import dataclass
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        from langchain_openai import ChatOpenAI
        from app.embeddings import get_bge_embeddings
    except Exception as exc:
        print("\nRAGAS evaluation is not available in this environment.")
        print(f"Reason: {exc}")
        print("Install the missing packages, then rerun:")
        print("  python -m pip install ragas datasets langchain-openai")
        return

    config = _ragas_config()
    if not _check_ollama_ready(config):
        return
    runtime_config = _ragas_runtime_config()
    print("  Runtime tuning:")
    _print_runtime_config(runtime_config)

    @dataclass
    class LocalOllamaFaithfulness(Faithfulness, LocalOllamaFaithfulnessMixin):
        async def _create_statements(self, row: Dict[str, Any], callbacks: Any) -> Any:
            return await self._generate_statements_with_retry(row, callbacks)

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "user_input": row["question"],
                "answer": row["answer"],
                "response": row["answer"],
                "contexts": row["contexts"],
                "retrieved_contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )

    base_evaluator_llm = ChatOpenAI(
        model=config["model"],
        temperature=0.0,
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=runtime_config["timeout"],
        max_retries=1,
    )
    run_config = RunConfig(
        timeout=runtime_config["timeout"],
        max_retries=2,
        max_workers=runtime_config["max_workers"],
    )
    evaluator_llm = LangchainLLMWrapper(base_evaluator_llm, run_config=run_config)
    evaluator_embeddings = get_bge_embeddings()
    faithfulness_metric = LocalOllamaFaithfulness()
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, answer_relevancy, context_precision],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            batch_size=1,
            raise_exceptions=False,
        )
    except Exception as exc:
        print("\nRAGAS evaluation failed after startup checks.")
        print(f"Reason: {exc}")
        print("This can happen when the local judge model is too slow, not chat-compatible enough for a metric, or Ollama times out.")
        _print_faithfulness_debug(faithfulness_metric)
        return
    result_dict = result.to_pandas().mean(numeric_only=True).to_dict()
    if not result_dict or all(
        value is None or (isinstance(value, float) and math.isnan(value))
        for value in result_dict.values()
    ):
        print("\nRAGAS evaluation completed, but the returned scores were empty or NaN.")
        print("This usually means the local judge failed to produce usable outputs for one or more metrics.")
        _print_faithfulness_debug(faithfulness_metric)
        _debug_faithfulness_probe(base_evaluator_llm, rows)
        return
    metric_rows = [
        {"metric": "faithfulness", "score": round(float(result_dict.get("faithfulness", 0.0)), 4)},
        {"metric": "answer_relevancy", "score": round(float(result_dict.get("answer_relevancy", 0.0)), 4)},
        {"metric": "context_precision", "score": round(float(result_dict.get("context_precision", 0.0)), 4)},
    ]

    print("\nRAGAS aggregate scores")
    _print_table(metric_rows, ["metric", "score"])
    faithfulness_score = result_dict.get("faithfulness")
    if faithfulness_score is None or (isinstance(faithfulness_score, float) and math.isnan(faithfulness_score)):
        print("\nFaithfulness could not be computed.")
        print("The other metrics completed, but the faithfulness prompt likely received output that the local judge did not format cleanly enough.")
        _print_faithfulness_debug(faithfulness_metric)
        _debug_faithfulness_probe(base_evaluator_llm, rows)


if __name__ == "__main__":
    main()
