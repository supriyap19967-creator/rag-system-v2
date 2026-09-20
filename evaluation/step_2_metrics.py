import os
import math
import pandas as pd
from typing import Any, Dict, List, Tuple
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, MLLMImage
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI

# 1. OpenRouter + Claude 3.5 Sonnet Integration
class OpenRouterLLM(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM class that routes all LLM-as-a-Judge evaluations
    through OpenRouter's API using anthropic/claude-3.5-sonnet.
    """
    def __init__(self, model_name: str = "anthropic/claude-3.5-sonnet"):
        self.model_name = model_name
        self.name = model_name
        self.supports_multimodal_attr = True
        # Set environment fallback for OPENAI_API_KEY if needed
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = "mock-key"
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def load_model(self):
        return self.client

    def supports_multimodal(self) -> bool:
        return True

    @property
    def model_name_prop(self) -> str:
        return self.model_name

    def _build_multimodal_messages(self, prompt: str) -> list[dict[str, Any]]:
        import re
        import base64
        
        # Check for custom visual context tags: [VISUAL_CONTEXT: /path/to/image.png]
        image_paths = re.findall(r'\[VISUAL_CONTEXT:\s*(.*?)\]', prompt)
        
        # Also scan for any inline deepeval MLLMImage representation or raw local image path
        if not image_paths:
            possible_paths = re.findall(r'([^\s\'"()]+?\.(?:png|jpg|jpeg|webp|gif))', prompt)
            for p in possible_paths:
                if os.path.exists(p):
                    image_paths.append(p)
                    
        if image_paths:
            content = []
            # Strip out visual tags/references from prompt to clean up textual query
            clean_prompt = re.sub(r'\[VISUAL_CONTEXT:\s*(.*?)\]', '', prompt).strip()
            content.append({"type": "text", "text": clean_prompt})
            
            for path in image_paths:
                path = path.strip()
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as img_file:
                            encoded = base64.b64encode(img_file.read()).decode('utf-8')
                        ext = os.path.splitext(path)[1][1:] or "png"
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{encoded}"
                            }
                        })
                    except Exception:
                        pass
            return [{"role": "user", "content": content}]
        
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str) -> str:
        messages = self._build_multimodal_messages(prompt)
        
        # DEBUG: Print Raw LLM Ingestion Request Payload (Summarized base64)
        print("\n" + "="*80)
        print("[DEBUG OpenRouterLLM Request]")
        print(f"Model: {self.model_name}")
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                print(f"Message [{idx}] ({role}):")
                for c in content:
                    c_type = c.get("type")
                    if c_type == "text":
                        print(f"  * Text: {c.get('text')[:300]}...")
                    elif c_type == "image_url":
                        url_val = c.get("image_url", {}).get("url", "")
                        print(f"  * Image: [Base64 String - Length: {len(url_val)} bytes]")
            else:
                print(f"Message [{idx}] ({role}): {str(content)[:300]}...")
        print("="*80 + "\n")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            res_content = response.choices[0].message.content
            print(f"[DEBUG OpenRouterLLM Response]: {res_content}\n")
            return res_content
        except Exception as e:
            print(f"Primary model {self.model_name} failed: {e}. Attempting fallback to google/gemini-2.5-flash...")
            fallback_model = "google/gemini-2.5-flash"
            response = self.client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                temperature=0.0
            )
            res_content = response.choices[0].message.content
            print(f"[DEBUG OpenRouterLLM Fallback Response]: {res_content}\n")
            return res_content

    async def a_generate(self, prompt: str) -> str:
        messages = self._build_multimodal_messages(prompt)
        
        # DEBUG: Print Raw LLM Ingestion Request Payload
        print("\n" + "="*80)
        print("[DEBUG OpenRouterLLM Async Request]")
        print(f"Model: {self.model_name}")
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                print(f"Message [{idx}] ({role}):")
                for c in content:
                    c_type = c.get("type")
                    if c_type == "text":
                        print(f"  * Text: {c.get('text')[:300]}...")
                    elif c_type == "image_url":
                        url_val = c.get("image_url", {}).get("url", "")
                        print(f"  * Image: [Base64 String - Length: {len(url_val)} bytes]")
            else:
                print(f"Message [{idx}] ({role}): {str(content)[:300]}...")
        print("="*80 + "\n")

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            res_content = response.choices[0].message.content
            print(f"[DEBUG OpenRouterLLM Async Response]: {res_content}\n")
            return res_content
        except Exception as e:
            print(f"Primary model {self.model_name} async failed: {e}. Attempting fallback to google/gemini-2.5-flash...")
            fallback_model = "google/gemini-2.5-flash"
            response = await self.async_client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                temperature=0.0
            )
            res_content = response.choices[0].message.content
            print(f"[DEBUG OpenRouterLLM Async Fallback Response]: {res_content}\n")
            return res_content

    def get_model_name(self) -> str:
        return self.model_name


# Define alias for DeepEvalBaseMetric
class DeepEvalBaseMetric(BaseMetric):
    """Base class for custom deterministic DeepEval metrics."""
    pass


# 2. Deterministic CSV Metric Engine
class DeterministicCSVMetric(DeepEvalBaseMetric):
    """
    A custom DeepEval metric for evaluating CSV/quantitative outputs using Pandas logic.
    Supports tolerance-based numeric comparisons and tabular matrix comparisons.
    """
    def __init__(self, expected_output: Any, threshold: float = 0.01):
        super().__init__()
        self.expected_output = expected_output
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def _parse_numeric(self, val: Any) -> float | None:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                clean_val = val.replace('$', '').replace(',', '').replace('%', '').strip()
                return float(clean_val)
            except ValueError:
                return None
        return None

    def _evaluate_numerical(self, predicted: Any, expected: Any) -> Tuple[float, bool, str]:
        pred_num = self._parse_numeric(predicted)
        exp_num = self._parse_numeric(expected)

        if pred_num is not None and exp_num is not None:
            diff = abs(pred_num - exp_num)
            is_match = diff <= self.threshold
            score = 1.0 if is_match else 0.0
            reason = (
                f"Numeric tolerance check passed: |{pred_num} - {exp_num}| <= {self.threshold}"
                if is_match else
                f"Numeric tolerance check failed: |{pred_num} - {exp_num}| > {self.threshold}"
            )
            return score, is_match, reason
        return 0.0, False, "Could not parse values for quantitative check."

    def _evaluate_tabular(self, predicted: Any, expected: Any) -> Tuple[float, bool, str]:
        def is_structured(v: Any) -> bool:
            if not isinstance(v, str):
                return True
            stripped = v.strip()
            return (stripped.startswith("{") or stripped.startswith("[")) or "|" in stripped

        try:
            if is_structured(expected) and is_structured(predicted):
                df_exp = None
                df_pred = None
                
                if isinstance(expected, str):
                    if "|" in expected:
                        df_exp = pd.read_csv(pd.compat.StringIO(expected), sep="|").dropna(how="all")
                    else:
                        df_exp = pd.read_json(expected)
                else:
                    df_exp = pd.DataFrame(expected)

                if isinstance(predicted, str):
                    if "|" in predicted:
                        df_pred = pd.read_csv(pd.compat.StringIO(predicted), sep="|").dropna(how="all")
                    else:
                        df_pred = pd.read_json(predicted)
                else:
                    df_pred = pd.DataFrame(predicted)

                if df_exp is not None and df_pred is not None:
                    overlap_cols = list(set(df_exp.columns) & set(df_pred.columns))
                    if not overlap_cols:
                        return 0.0, False, "No overlapping columns found for tabular assertion."
                    
                    matches = 0
                    total = 0
                    for col in overlap_cols:
                        for i in range(min(len(df_exp), len(df_pred))):
                            val_exp = df_exp[col].iloc[i]
                            val_pred = df_pred[col].iloc[i]
                            
                            num_exp = self._parse_numeric(val_exp)
                            num_pred = self._parse_numeric(val_pred)
                            if num_exp is not None and num_pred is not None:
                                if abs(num_exp - num_pred) <= self.threshold:
                                    matches += 1
                            elif str(val_exp).strip().lower() == str(val_pred).strip().lower():
                                matches += 1
                            total += 1
                    
                    match_rate = matches / total if total > 0 else 0.0
                    is_match = match_rate >= 0.8
                    return match_rate, is_match, f"Tabular match rate: {match_rate:.2%} ({matches}/{total} cells matched)"
        except Exception as e:
            pass

        # Fallback string comparison
        str_exp = str(expected).strip().lower()
        str_pred = str(predicted).strip().lower()
        if str_exp in str_pred or str_pred in str_exp:
            return 1.0, True, "Tabular match via substring containment."
        return 0.0, False, f"Tabular exact and substring checks failed. Expected: '{expected}', Predicted: '{predicted}'"

    def measure(self, test_case: LLMTestCase) -> float:
        pred = test_case.actual_output
        exp = self.expected_output

        is_num = (self._parse_numeric(pred) is not None) and (self._parse_numeric(exp) is not None)
        if is_num:
            self.score, self.success, self.reason = self._evaluate_numerical(pred, exp)
        else:
            self.score, self.success, self.reason = self._evaluate_tabular(pred, exp)
            
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Deterministic CSV Metric"


# 3. Hybrid Evaluator Router
def evaluate_sample(sample: Dict[str, Any], test_case: LLMTestCase) -> Dict[str, Any]:
    """
    Master routing function that evaluates a test case dynamically based on its modality tag.
    Combines deterministic CSV checks and Vision-Language GEval checks.
    """
    modalities = sample.get("modality", [])
    expected = sample.get("expected_output")
    inputs = sample.get("inputs", {})
    
    scores = []
    reasons = []
    
    # Instantiate the custom OpenRouter LLM judge
    judge_llm = OpenRouterLLM()

    # Case A: CSV Metric Check
    if "csv" in modalities:
        csv_metric = DeterministicCSVMetric(expected_output=expected)
        csv_score = csv_metric.measure(test_case)
        scores.append(csv_score)
        reasons.append(f"[CSV Check] Score: {csv_score:.2f} | Reason: {csv_metric.reason}")

    # Case B: Visual Check
    if "visual" in modalities:
        image_path = inputs.get("image_path")
        print(f"[PATH CHECK] CWD: {os.getcwd()} | Original Image Path: {image_path} | Exists: {os.path.exists(image_path) if image_path else False}")
        if image_path:
            # Attempt to resolve relative to project root or evaluation folder if needed
            if not os.path.exists(image_path):
                # Try prefixing with project root
                from pathlib import Path
                alt_path = str(Path(__file__).resolve().parent.parent / image_path)
                print(f"[PATH CHECK] Attempting alt path: {alt_path} | Exists: {os.path.exists(alt_path)}")
                if os.path.exists(alt_path):
                    image_path = alt_path
            
        if image_path and os.path.exists(image_path):
            image_obj = MLLMImage(url=image_path, local=True)
            
            # Enforce Dual Representation: Visual Reference tag + Text Metadata
            dual_retrieved_contexts = [
                f"[VISUAL_CONTEXT: {image_path}]",
                f"[Visual Metadata]: Figure caption and layout details: {expected}"
            ]
            
            # Re-construct visual-specific test case to embed MLLMImage context and dual retrieved_contexts
            visual_test_case = LLMTestCase(
                input=f"{test_case.input} [VISUAL_CONTEXT: {image_path}] [Context Image: {image_obj}]",
                actual_output=test_case.actual_output,
                expected_output=str(expected),
                retrieved_contexts=dual_retrieved_contexts
            )
            
            # VLM visual GEval metric setup
            visual_metric = GEval(
                name="Visual Extraction Accuracy",
                criteria=(
                    "Evaluate whether the visual data, layout, or chart properties extracted from the image "
                    "exactly match the expected output and are correctly reflected in the predicted output."
                ),
                model=judge_llm,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT
                ]
            )
            
            visual_score = visual_metric.measure(visual_test_case)
            scores.append(visual_score)
            reasons.append(f"[Visual Check] Score: {visual_score:.2f} | Reason: {visual_metric.reason}")
        else:
            scores.append(0.0)
            reasons.append(f"[Visual Check] Failed: Asset path '{image_path}' not found.")

    # Case C: Text-only (Fallback or Explicit)
    if not scores or ("text" in modalities and len(modalities) == 1):
        text_metric = GEval(
            name="Answer Correctness",
            criteria="Determine if the predicted output semantically matches the expected output.",
            model=judge_llm,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ]
        )
        text_score = text_metric.measure(test_case)
        scores.append(text_score)
        reasons.append(f"[Text Check] Score: {text_score:.2f} | Reason: {text_metric.reason}")

    # Consolidate hybrid metrics
    aggregated_score = sum(scores) / len(scores) if scores else 0.0
    aggregated_reason = " | ".join(reasons)

    return {
        "score": round(aggregated_score, 4),
        "reason": aggregated_reason
    }

# Register OpenRouterLLM as a supported multimodal model inside DeepEval's utilities
try:
    from deepeval.metrics.utils import MULTIMODAL_SUPPORTED_MODELS
    MULTIMODAL_SUPPORTED_MODELS[OpenRouterLLM] = {
        "anthropic/claude-3.5-sonnet": lambda *args, **kwargs: OpenRouterLLM(*args, **kwargs)
    }
except Exception:
    pass
