#!/usr/bin/env python
"""
Offline Batch Evaluation Script.
Fetches recent traces from Langfuse, identifies unevaluated runs,
runs Ragas evaluations sequentially with rate limiting, and posts the scores back.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("offline_evaluator")

from langfuse import Langfuse
from app.evaluators.ragas_langfuse_evaluator import run_ragas_evaluation

def run_batch_eval():
    logger.info("Starting offline batch evaluation...")
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        logger.error("Missing Langfuse API credentials in environment variables.")
        return
        
    langfuse_client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )
    
    logger.info("Fetching recent traces from Langfuse...")
    try:
        if hasattr(langfuse_client, "api") and hasattr(langfuse_client.api, "trace"):
            traces = langfuse_client.api.trace.list(limit=50).data
        elif hasattr(langfuse_client, "get_traces"):
            traces = langfuse_client.get_traces(limit=50).data
        else:
            logger.error("Unable to locate trace listing method on Langfuse client.")
            return
        logger.info(f"Retrieved {len(traces)} traces from Langfuse.")
    except Exception as exc:
        logger.exception(f"Failed to fetch traces from Langfuse: {exc}")
        return

    evaluated_count = 0
    
    for trace in traces:
        trace_id = trace.id
        
        query = trace.input
        answer = trace.output
        
        if not query or not answer:
            continue
            
        query = str(query).strip()
        answer = str(answer).strip()
        
        existing_scores = []
        for s in getattr(trace, "scores", []):
            if isinstance(s, str):
                existing_scores.append(s)
            elif hasattr(s, "name"):
                existing_scores.append(s.name)
            elif isinstance(s, dict) and "name" in s:
                existing_scores.append(s["name"])
        try:
            if hasattr(langfuse_client, "api") and hasattr(langfuse_client.api, "scores"):
                scores_data = langfuse_client.api.scores.get_many(trace_id=trace_id).data
                for s in scores_data:
                    existing_scores.append(s.name)
        except Exception:
            pass
                
        if any(m in existing_scores for m in ["answer_relevancy", "context_precision", "llm_context_precision_without_reference"]):
            logger.info(f"Trace {trace_id} is already evaluated. Skipping.")
            continue
            
        logger.info(f"Processing evaluation for trace {trace_id}...")
        
        metadata = getattr(trace, "metadata", {}) or {}
        text_chunks = metadata.get("retrieved_contexts", [])
        csv_rows = metadata.get("csv_contexts", [])
        
        try:
            asyncio.run(run_ragas_evaluation(
                trace_id=trace_id,
                query=query,
                answer=answer,
                text_chunks=text_chunks,
                csv_rows=csv_rows,
                ocr_chart_data=[]
            ))
            evaluated_count += 1
            logger.info(f"Successfully evaluated trace {trace_id}.")
            time.sleep(1.5)
        except Exception as eval_exc:
            logger.error(f"Failed to evaluate trace {trace_id}: {eval_exc}")

    logger.info(f"Batch evaluation finished. Evaluated {evaluated_count} new traces.")

if __name__ == "__main__":
    run_batch_eval()
