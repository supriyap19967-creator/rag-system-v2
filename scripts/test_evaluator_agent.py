from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.evaluator import execute_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_evaluator() -> None:
    logger.info("Executing Evaluator Agent test...")
    user_query = "What is the average GDP for India?"
    retrieved_context = "Country Name,Year,GDP\nIndia,2019,2835606256558.19\nIndia,2020,2660298248696.83"
    
    # 1. Test faithful response
    logger.info("="*60)
    logger.info("Testing Faithful Response:")
    agent_response_1 = "**📊 India average GDP (2019-2020)**\n* **Average GDP:** $2.75 Trillion ($2,747,952,252,627.51)\n\n*Source: Consolidated pandas query.*"
    try:
        res1 = execute_evaluation(user_query, retrieved_context, agent_response_1)
        logger.info(f"Is Faithful: {res1['is_faithful']}")
        logger.info(f"Score: {res1['score']}")
        logger.info(f"Reasoning: {res1['reasoning']}")
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    # 2. Test hallucinated response
    logger.info("="*60)
    logger.info("Testing Hallucinated Response:")
    agent_response_2 = "**📊 India average GDP (2019-2020)**\n* **Average GDP:** $5.80 Trillion ($5,800,000,000,000.00)\n\n*Source: Consolidated pandas query.*"
    try:
        res2 = execute_evaluation(user_query, retrieved_context, agent_response_2)
        logger.info(f"Is Faithful: {res2['is_faithful']}")
        logger.info(f"Score: {res2['score']}")
        logger.info(f"Reasoning: {res2['reasoning']}")
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")
    logger.info("="*60)

if __name__ == "__main__":
    test_evaluator()
