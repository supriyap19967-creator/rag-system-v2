import os
import json
import uuid
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
from app.embeddings import get_bge_embeddings

load_dotenv()

API_URL = "http://127.0.0.1:8000/query"
LOG_FILE = "evaluation_logs.json"

GOLDEN_TEST_SET = [
    {
        "question": "What was India's GDP in 2022 according to the CSV data?",
        "type": "NUMERICAL"
    },
    {
        "question": "What are the key regulations for 'nation building' mentioned in the documents?",
        "type": "QUALITATIVE"
    },
    {
        "question": "How do financial standards affect GDP growth in emerging economies?",
        "type": "HYBRID"
    }
]

def run_evaluation(disable_intent_routing=False):
    results = []
    for item in GOLDEN_TEST_SET:
        print(f"Running query: {item['question']} (Routing: {not disable_intent_routing})")
        try:
            response = requests.post(
                API_URL,
                json={
                    "session_id": str(uuid.uuid4()),
                    "question": item["question"],
                    "disable_intent_routing": disable_intent_routing
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            results.append({
                "question": item["question"],
                "answer": data.get("answer", ""),
                "contexts": data.get("contexts", []),
                "type": item["type"]
            })
        except Exception as e:
            print(f"Error running query: {e}")
    
    if not results:
        return None

    dataset = Dataset.from_list(results)
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_bge_embeddings())

    eval_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    return eval_result.to_pandas()

def main():
    print("=== Starting Ragas Evaluation Pipeline ===")
    
    # Check if API is running
    try:
        requests.get("http://127.0.0.1:8000/health")
    except:
        print("Error: API is not running at http://127.0.0.1:8000. Please start it first.")
        return

    print("\n--- Running Baseline (Intent Routing Disabled) ---")
    baseline_df = run_evaluation(disable_intent_routing=True)
    
    print("\n--- Running Optimized (Intent Routing Enabled) ---")
    optimized_df = run_evaluation(disable_intent_routing=False)
    
    if baseline_df is None or optimized_df is None:
        print("Evaluation failed.")
        return

    baseline_scores = baseline_df.mean(numeric_only=True).to_dict()
    optimized_scores = optimized_df.mean(numeric_only=True).to_dict()

    summary = {
        "baseline": baseline_scores,
        "optimized": optimized_scores,
        "improvement": {
            metric: ((optimized_scores[metric] - baseline_scores[metric]) / baseline_scores[metric] * 100)
            if baseline_scores[metric] != 0 else 0
            for metric in baseline_scores
        }
    }

    with open(LOG_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=== Evaluation Results ===")
    print(f"Baseline Faithfulness: {baseline_scores['faithfulness']:.4f}")
    print(f"Optimized Faithfulness: {optimized_scores['faithfulness']:.4f}")
    
    f_imp = summary["improvement"]["faithfulness"]
    if f_imp > 0:
        print(f"Optimization Success: Intent Routing improved faithfulness by {f_imp:.2f}%.")
    else:
        print(f"Optimization Note: Faithfulness change: {f_imp:.2f}%.")

    print(f"\nFull results saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
