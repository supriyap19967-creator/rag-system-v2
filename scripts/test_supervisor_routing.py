from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.supervisor import orchestrate_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_routing() -> None:
    test_queries = [
        "What is the average GDP growth rate from Table 4.1?",
        "Can you display the chart from Figure 4.1 and extract its values?",
        "What does Chapter 5 say about the impact of inflation?",
        "Who wrote the book War and Peace?"
    ]
    
    logger.info("Starting Supervisor routing tests...")
    
    for idx, query in enumerate(test_queries, 1):
        logger.info("="*60)
        logger.info(f"Test case {idx}: {query}")
        try:
            decision = orchestrate_query(query)
            logger.info("Routing Result:")
            logger.info(f"  Target Agent: {decision['routing_decision']}")
            logger.info(f"  Reasoning: {decision['reasoning']}")
            logger.info(f"  Sub-queries: {decision['sub_queries']}")
            logger.info(f"  Validation Required: {decision['requires_validation']}")
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            
    logger.info("="*60)
    logger.info("Supervisor routing tests completed.")

if __name__ == "__main__":
    test_routing()
