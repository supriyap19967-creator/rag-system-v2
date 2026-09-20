from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.data import execute_data_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_data() -> None:
    logger.info("Executing Data Agent test query...")
    query = "Print the columns and the first 3 rows of the preloaded DataFrame variable 'page_206_Table_4_1'."
    
    try:
        result = execute_data_task(query)
        logger.info("="*60)
        logger.info("DATA AGENT OUTPUT:")
        logger.info(f"Executive Metrics:\n{result['executive_metrics']}")
        logger.info(f"\nAnalytical Breakdown:\n{result['analytical_breakdown']}")
        logger.info(f"\nCode Reference:\n{result['code_reference']}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Data test execution failed: {e}")

if __name__ == "__main__":
    test_data()
