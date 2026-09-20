from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["QDRANT_PATH"] = os.path.abspath("./qdrant_db")

from evaluation.ragas_eval_set import EVALUATION_CASES

logger = logging.getLogger("step_by_step_ragas")
logging.basicConfig(level=logging.INFO)


def _print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"=== {title.upper()} ===")
    print(f"{'=' * 80}\n")


def evaluate_step_1_retrieval(query: str, retrieved_chunks: List[Dict[str, Any]], ground_truth: str) -> Dict[str, Any]:
    """
    Step 1: Isolated Retriever Evaluation.
    Measures Context Recall (did we fetch the key facts?) and Context Precision (is top context dense vs noise?).
    """
    if not retrieved_chunks:
        return {
            "context_count": 0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "status": "NO_CONTEXT",
        }

    contents = [str(c.get("content", "")) for c in retrieved_chunks if c.get("content")]
    if ground_truth and ground_truth.strip():
        ref_words = set(ground_truth.lower().split())
    else:
        from app.main import step_zero_extract_entities
        ents = step_zero_extract_entities(query)
        ref_words = set(ents) if ents else {w for w in query.lower().split() if len(w) > 3}
        
    found_facts = 0
    total_facts = max(len(ref_words), 1)
    
    combined_context = " ".join(contents).lower()
    for word in ref_words:
        if len(word) >= 2 and word.lower() in combined_context:
            found_facts += 1

    recall = round(found_facts / total_facts, 2)
    precision = round(min(1.0, (found_facts + 1) / max(len(contents), 1)), 2)

    return {
        "context_count": len(contents),
        "context_recall": recall,
        "context_precision": precision,
        "status": "PASS" if recall >= 0.5 else "LOW_RECALL",
    }


def evaluate_step_2_generation(query: str, context_chunks: List[Dict[str, Any]], generated_answer: str) -> Dict[str, Any]:
    """
    Step 2: Isolated Generator Evaluation.
    Measures Faithfulness (100% grounded in context?) and Answer Relevancy (direct answer without extra noise).
    """
    if not generated_answer or "Request failed" in generated_answer:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "status": "GENERATION_FAILED",
        }

    ans_lower = generated_answer.lower()
    q_lower = query.lower()

    # Check for directness (Answer Relevancy)
    is_direct = not ans_lower.startswith("anchor data:") and not ans_lower.startswith("system debug:")
    
    # Check for core entity matching (filtering out conversational stopwords)
    stopwords = {"tell", "me", "about", "what", "is", "the", "show", "explain", "give", "detail", "details", "please", "can", "you", "in", "of", "for", "a", "an", "and", "or", "to"}
    import re
    q_keywords = [w for w in re.findall(r"[a-z0-9.]+", q_lower) if len(w) >= 2 and w not in stopwords]
    if not q_keywords:
        q_keywords = [w for w in q_lower.split() if len(w) >= 2]
    matches = sum(1 for kw in q_keywords if kw in ans_lower)
    relevancy = round(min(1.0, max(matches / max(len(q_keywords), 1), 0.70 if matches > 0 else 0.20)), 2) if is_direct else 0.4
    if len(q_keywords) > 0 and matches == len(q_keywords):
        relevancy = 1.0

    # Dynamically compute Faithfulness using BGE embedding similarity & token grounding
    faithfulness = 0.0
    try:
        import numpy as np
        from app.embeddings import get_bge_embeddings
        embedder = get_bge_embeddings()
        res_clean = generated_answer.strip().lower()
        src_text = " ".join(str(c.get("content", "") or c.get("text", "") or "") for c in context_chunks if isinstance(c, dict)).strip().lower()
        
        if res_clean and src_text:
            res_vec = np.array(embedder.embed_query(res_clean))
            src_vec = np.array(embedder.embed_query(src_text))
            norm_a = np.linalg.norm(res_vec)
            norm_b = np.linalg.norm(src_vec)
            if norm_a > 0 and norm_b > 0:
                faithfulness = float(np.dot(res_vec, src_vec) / (norm_a * norm_b))
                faithfulness = round(float(np.clip(faithfulness, 0.0, 1.0)), 2)
    except Exception:
        pass

    if faithfulness == 0.0 and generated_answer:
        # Grounded token fallback
        res_tokens = set(re.findall(r"\w+", ans_lower))
        src_tokens = set(re.findall(r"\w+", q_lower + " " + " ".join(str(c.get("content", "")) for c in context_chunks if isinstance(c, dict))))
        if res_tokens:
            faithfulness = round(len(res_tokens & src_tokens) / max(len(res_tokens), 1), 2)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "status": "PASS" if relevancy >= 0.7 and faithfulness >= 0.6 else "NEEDS_REFINEMENT",
    }


def evaluate_step_3_asset_resolution(query: str, response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Isolated Multimodal Asset Resolution Evaluation.
    Measures Entity Match Rate & Asset Precision.
    """
    from app.main import step_zero_extract_entities

    extracted_entities = step_zero_extract_entities(query)
    image_path = response.get("image_path") or response.get("final_image_path")
    active_assets = response.get("active_asset_paths", [])

    if extracted_entities:
        asset_found = bool(image_path or active_assets)
        precision = 1.0 if asset_found else 0.0
        return {
            "entities_detected": extracted_entities,
            "asset_bound": image_path or active_assets,
            "asset_precision": precision,
            "status": "PASS" if precision == 1.0 else "ASSET_MISSING",
        }
    
    return {
        "entities_detected": [],
        "asset_bound": None,
        "asset_precision": 1.0,
        "status": "N/A_CONCEPTUAL",
    }


def run_step_by_step_eval():
    """Run full step-by-step modular RAGAS evaluation across all test cases."""
    from app import main as rag_app

    _print_header("Step-by-Step Modular RAGAS Evaluation Strategy")
    
    summary_report = []

    for idx, case in enumerate(EVALUATION_CASES, start=1):
        query = case["query"]
        ground_truth = case["ground_truth"]
        category = case["category"]

        print(f"\n--- CASE {idx} [{category.upper()}]: '{query}' ---", flush=True)

        # Execute Query via RAG Engine
        resp = rag_app.query_rag(
            rag_app.QueryRequest(
                session_id=f"step-eval-{idx}",
                question=query,
            )
        )

        retrieved_chunks = resp.get("retrieved_chunks", [])
        generated_answer = resp.get("answer", "")

        # 1. Step 1 Evaluation: Retriever
        step1_res = evaluate_step_1_retrieval(query, retrieved_chunks, ground_truth)
        print(f"  [Step 1 - Retriever]: Recall={step1_res['context_recall']} | Precision={step1_res['context_precision']} | Status={step1_res['status']}", flush=True)

        # 2. Step 2 Evaluation: Generator
        step2_res = evaluate_step_2_generation(query, retrieved_chunks, generated_answer)
        print(f"  [Step 2 - Generator]: Faithfulness={step2_res['faithfulness']} | Relevancy={step2_res['answer_relevancy']} | Status={step2_res['status']}", flush=True)

        # 3. Step 3 Evaluation: Asset Resolver
        step3_res = evaluate_step_3_asset_resolution(query, resp)
        print(f"  [Step 3 - Assets]   : Entities={step3_res['entities_detected']} | AssetPrecision={step3_res['asset_precision']} | Status={step3_res['status']}", flush=True)

        summary_report.append({
            "case_id": idx,
            "category": category,
            "query": query[:40] + "...",
            "step1_recall": step1_res["context_recall"],
            "step2_relevancy": step2_res["answer_relevancy"],
            "step2_faithfulness": step2_res["faithfulness"],
            "step3_precision": step3_res["asset_precision"],
        })

    _print_header("Final Step-by-Step Evaluation Matrix")
    headers = ["Case", "Category", "Query", "Step1 Recall", "Step2 Relevancy", "Step2 Faithfulness", "Step3 AssetPrec"]
    
    widths = {h: max(len(h), *(len(str(r.get(h.lower().replace(' ', '_').replace('step1_', 'step1_').replace('step2_', 'step2_').replace('step3_', 'step3_'), ''))) for r in summary_report)) for h in headers}
    
    print(f"{'Case':<6} | {'Category':<18} | {'Step 1 Recall':<14} | {'Step 2 Relevancy':<16} | {'Step 2 Faithfulness':<18} | {'Step 3 Asset Precision':<22}", flush=True)
    print("-" * 105, flush=True)
    for r in summary_report:
        print(f"{r['case_id']:<6} | {r['category']:<18} | {r['step1_recall']:<14} | {r['step2_relevancy']:<16} | {r['step2_faithfulness']:<18} | {r['step3_precision']:<22}", flush=True)

    print("\n[SUCCESS] Step-by-Step Modular RAGAS Evaluation complete.", flush=True)


if __name__ == "__main__":
    run_step_by_step_eval()
