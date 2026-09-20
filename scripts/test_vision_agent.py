from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.agents.vision import execute_vision_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_vision() -> None:
    logger.info("Executing Vision Agent test query...")
    image_path = "assets/extracted_images/page_206_Figure_4.1.png"
    query = "What is the trend shown in the chart?"
    
    try:
        result = execute_vision_task(image_path, query)
        logger.info("="*60)
        logger.info("VISION AGENT OUTPUT:")
        logger.info(f"Visual Summary:\n{result['visual_summary']}")
        logger.info("\nVisual Analysis & Breakdown:")
        for bullet in result['visual_analysis']:
            logger.info(f"  - {bullet}")
        logger.info(f"\nActionable Insights:\n{result['actionable_insights']}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Vision test execution failed: {e}")

if __name__ == "__main__":
    test_vision()
