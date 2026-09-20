from __future__ import annotations

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.evaluators.ragas_langfuse_evaluator import run_ragas_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ragas() -> None:
    logger.info("Executing Ragas Evaluation test...")
    trace_id = "test-trace-id-123"
    query = "What is the average GDP for India?"
    answer = "The average GDP is $2.75 Trillion ($2,747,952,252,627.51)."
    text_chunks = ["India,2019,2835606256558.19", "India,2020,2660298248696.83"]
    
    try:
        asyncio.run(run_ragas_evaluation(
            trace_id=trace_id,
            query=query,
            answer=answer,
            text_chunks=text_chunks,
            csv_rows=[],
            ocr_chart_data=[]
        ))
    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")

if __name__ == "__main__":
    test_ragas()
