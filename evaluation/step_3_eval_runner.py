import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from deepeval.test_case import LLMTestCase
from evaluation.step_1_dataset import multimodal_eval_dataset
from evaluation.step_2_metrics import evaluate_sample

def mock_multimodal_system(sample: Dict[str, Any]) -> Any:
    """
    Mock runner simulating the multimodal system's actual_output generation.
    Returns realistic text, tabular CSV, or visual answers.
    """
    modality = sample.get("modality", [])
    expected = sample.get("expected_output")
    
    # Simulate realistic generations
    if sample["id"] == "eval-001":  # text-only
        return "Standards lower transaction costs, signal product quality to international buyers, and facilitate the transfer of best-practice knowledge."
    elif sample["id"] == "eval-002":  # csv-only
        # Return exact value
        return 2747952252627.51
    elif sample["id"] == "eval-003":  # visual-only
        # Return a slightly modified version to test metric robustness
        return "Figure 5.1 shows that vaccine coverage has increased significantly while the under-five mortality rate has steadily declined between 1974 and 2023."
    elif sample["id"] == "eval-004":  # text + csv
        # Return exact matching statement
        return "The difference is 12% and the variation is due to compliance costs and capacity limitations in lower-income countries."
    elif sample["id"] == "eval-005":  # text + visual
        return "Standard harmonization reduces trade barriers, complementing the coexistence of environmental standards and other policy instruments shown in Figure 6.1."
    elif sample["id"] == "eval-006":  # csv + visual
        return "The CSV emissions data indicates rising trends for India, while Table 4.1 outlines firm-level impacts of standards adoption, showing positive outcomes across studies for sales growth, profit growth, and exports."
    elif sample["id"] == "eval-007":  # text + csv + visual
        return "Place-based localized air pollution rules can trigger emissions leakage, which can be addressed by coexisting policy instruments such as those illustrated in Figure 6.1's network."
        
    return str(expected)

def run_evaluation_pipeline() -> List[Dict[str, Any]]:
    """
    Runs the offline pipeline execution engine.
    Iterates over the evaluation dataset, runs the system mock, evaluates using DeepEval metrics,
    and consolidates structured execution metadata ready for trace ingestion.
    """
    execution_results = []
    
    print("\n" + "="*80)
    print("STARTING MULTIMODAL PIPELINE EVALUATION RUNNER (STEP 3)")
    print("="*80 + "\n")
    
    for sample in multimodal_eval_dataset:
        sample_id = sample["id"]
        modality = sample["modality"]
        question = sample["question"]
        
        print(f"Processing sample {sample_id} | Modalities: {modality}")
        print(f"Question: '{question}'")
        
        # 1. Run the system mock to produce a predicted output
        start_time = time.time()
        actual_output = mock_multimodal_system(sample)
        latency = time.time() - start_time
        
        # 2. Evaluate sample output using custom DeepEval routing logic
        print("Running DeepEval evaluation routing...")
        test_case = LLMTestCase(
            input=question,
            actual_output=str(actual_output),
            expected_output=str(sample.get("expected_output"))
        )
        eval_metadata = evaluate_sample(sample, test_case)
        score = eval_metadata["score"]
        passed = score >= 0.8
        reason = eval_metadata["reason"]
        
        # 3. Compile structured results
        result = {
            "id": sample_id,
            "question": question,
            "modality": modality,
            "inputs": sample["inputs"],
            "expected_output": sample["expected_output"],
            "actual_output": actual_output,
            "score": score,
            "passed": passed,
            "explanation": reason,
            "metadata": {
                "latency_seconds": round(latency, 4),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }
        execution_results.append(result)
        
        print(f"Result -> Score: {result['score']} | Status: {'PASS' if result['passed'] else 'FAIL'}")
        print(f"Explanation:\n{result['explanation']}")
        print("-"*80 + "\n")
        
    print("="*80)
    print(f"EVALUATION PIPELINE COMPLETED: {len(execution_results)} samples processed.")
    print("="*80 + "\n")
    
    return execution_results

if __name__ == "__main__":
    run_evaluation_pipeline()
