import os
import math
import pandas as pd
from typing import Any, Dict, List, Tuple
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, MLLMImage

# Define alias for DeepEvalBaseMetric to satisfy requirements
class DeepEvalBaseMetric(BaseMetric):
    """Base class for custom deterministic DeepEval metrics."""
    pass

class DeterministicCSVMetric(DeepEvalBaseMetric):
    """
    A deterministic DeepEval metric for evaluating CSV/quantitative outputs using Pandas logic.
    Handles exact numerical matching (with tolerance) and tabular value matching.
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
                # Remove currency symbols, commas, percent signs, and whitespace
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
                f"Numerical check succeeded: |{pred_num} - {exp_num}| <= {self.threshold}"
                if is_match else
                f"Numerical check failed: |{pred_num} - {exp_num}| > {self.threshold}"
            )
            return score, is_match, reason
        return 0.0, False, "Could not parse predicted or expected outputs as numerical values."

    def _evaluate_tabular(self, predicted: Any, expected: Any) -> Tuple[float, bool, str]:
        try:
            # Try parsing the expected and predicted values as DataFrames or dictionary objects
            if isinstance(expected, str):
                try:
                    # Look for markdown or json-like structures
                    if "|" in expected:
                        df_exp = pd.read_csv(pd.compat.StringIO(expected), sep="|").dropna(how="all")
                    else:
                        df_exp = pd.read_json(expected)
                except Exception:
                    df_exp = None
            else:
                df_exp = pd.DataFrame(expected)

            if isinstance(predicted, str):
                try:
                    if "|" in predicted:
                        df_pred = pd.read_csv(pd.compat.StringIO(predicted), sep="|").dropna(how="all")
                    else:
                        df_pred = pd.read_json(predicted)
                except Exception:
                    df_pred = None
            else:
                df_pred = pd.DataFrame(predicted)

            if df_exp is not None and df_pred is not None:
                # Align columns and rows to check overlapping elements
                overlap_cols = list(set(df_exp.columns) & set(df_pred.columns))
                if not overlap_cols:
                    return 0.0, False, "Tabular alignment failed: No matching columns found between predicted and expected tables."
                
                # Check match rate on aligned values
                matches = 0
                total = 0
                for col in overlap_cols:
                    for i in range(min(len(df_exp), len(df_pred))):
                        val_exp = df_exp[col].iloc[i]
                        val_pred = df_pred[col].iloc[i]
                        
                        # Numerical check within table cell
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
                return match_rate, is_match, f"Tabular match rate: {match_rate:.2%} (aligned cells: {matches}/{total})"
            
        except Exception as e:
            return 0.0, False, f"Error processing tabular check: {str(e)}"

        # Fallback to string matching
        str_pred = str(predicted).strip().lower()
        str_exp = str(expected).strip().lower()
        if str_exp in str_pred or str_pred in str_exp:
            return 1.0, True, "Tabular fallback substring match succeeded."
        return 0.0, False, "Tabular exact and substring comparisons failed."

    def measure(self, test_case: LLMTestCase) -> float:
        pred = test_case.actual_output
        exp = self.expected_output

        # Route dynamically depending on contents
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


def evaluate_sample(sample: Dict[str, Any], predicted_output: Any) -> Dict[str, Any]:
    """
    Consolidated routing function that applies relevant metrics based on the modality tag:
    - Runs DeterministicCSVMetric if 'csv' is in modality list.
    - Runs GEval (VLM powered) if 'visual' is in modality list.
    - Runs basic GEval correctness if only 'text' is present.
    """
    modalities = sample.get("modality", [])
    inputs = sample.get("inputs", {})
    expected = sample.get("expected_output")
    question = sample.get("question", "")

    scores = []
    explanations = []

    # 1. Route to CSV Deterministic evaluation
    if "csv" in modalities:
        csv_metric = DeterministicCSVMetric(expected_output=expected)
        test_case = LLMTestCase(
            input=question,
            actual_output=str(predicted_output)
        )
        csv_score = csv_metric.measure(test_case)
        scores.append(csv_score)
        explanations.append(f"[CSV Check] Score: {csv_score:.2f}. Reason: {csv_metric.reason}")

    # 2. Route to Visual / Vision evaluation
    if "visual" in modalities:
        image_path = inputs.get("image_path")
        if image_path and os.path.exists(image_path):
            # Create a test case with the image
            image_obj = MLLMImage(url=image_path, local=True)
            
            # GEval configured for multimodal visual inspection
            visual_metric = GEval(
                name="Visual Extraction Accuracy",
                criteria="Evaluate whether the visual data extracted from the image/chart aligns exactly with the expected_output and predicted_output.",
                model="gpt-4o",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT
                ]
            )
            
            test_case = LLMTestCase(
                input=f"Chart Image context: {question}",
                actual_output=f"Extracted content: {predicted_output} {image_obj}",
                expected_output=str(expected)
            )
            
            visual_score = visual_metric.measure(test_case)
            scores.append(visual_score)
            explanations.append(f"[Visual Check] Score: {visual_score:.2f}. Reason: {visual_metric.reason}")
        else:
            scores.append(0.0)
            explanations.append(f"[Visual Check] Failed: Visual asset path '{image_path}' not found or invalid.")

    # 3. Fallback for pure Text-based evaluation if no other modal metrics ran
    if not scores or ("text" in modalities and len(modalities) == 1):
        text_metric = GEval(
            name="Answer Correctness",
            criteria="Determine if the actual output matches the semantic meaning of the expected output.",
            model="gpt-4o",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ]
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=str(predicted_output),
            expected_output=str(expected)
        )
        text_score = text_metric.measure(test_case)
        scores.append(text_score)
        explanations.append(f"[Text Check] Score: {text_score:.2f}. Reason: {text_metric.reason}")

    # Consolidate results
    final_score = sum(scores) / len(scores) if scores else 0.0
    passed = final_score >= 0.8  # Pass threshold
    detailed_explanation = "\n".join(explanations)

    return {
        "id": sample.get("id"),
        "modalities": modalities,
        "score": round(final_score, 4),
        "passed": passed,
        "explanation": detailed_explanation
    }
