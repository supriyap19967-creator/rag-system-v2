from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.validation import execute_validation_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_validation() -> None:
    logger.info("Executing Validation Agent test query...")
    user_query = "What is the trend shown in Figure 4.1?"
    agent_text_response = "The chart shows a positive correlation between internationally recognized standards and GDP per capita."
    visual_asset_path = "assets/extracted_images/page_206_Figure_4.1.png"
    
    try:
        result = execute_validation_task(user_query, agent_text_response, visual_asset_path)
        logger.info("="*60)
        logger.info("VALIDATION AGENT OUTPUT:")
        logger.info(f"Audit Summary: {result['audit_summary']}")
        logger.info(f"\nRendered Card:\n{result['card_payload']}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Validation test execution failed: {e}")

if __name__ == "__main__":
    test_validation()
