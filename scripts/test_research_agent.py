from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.research import execute_research

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_research() -> None:
    logger.info("Executing Research Agent test query...")
    query = "What does the report say about standard adopted and inflation?"
    try:
        result = execute_research(query)
        logger.info("="*60)
        logger.info("RESEARCH AGENT OUTPUT:")
        logger.info(f"Executive Summary:\n{result['executive_summary']}")
        logger.info("\nKey Findings:")
        for finding in result['key_findings']:
            logger.info(f"  - {finding}")
        logger.info(f"\nSource Notes:\n{result['source_notes']}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Research test execution failed: {e}")

if __name__ == "__main__":
    test_research()
