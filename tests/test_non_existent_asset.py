import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from streamlit_ui.StreamlitApp import is_target_asset_existing, process_vision_element


class TestNonExistentAssetFix(unittest.TestCase):

    def test_non_existent_asset_detection(self):
        # Table 6.1 does not exist in dataset
        self.assertFalse(is_target_asset_existing("Table", "6.1"))
        # Figure 9.9 does not exist in dataset
        self.assertFalse(is_target_asset_existing("Figure", "9.9"))

    def test_existing_asset_detection(self):
        # Table 3.1 exists in dataset
        self.assertTrue(is_target_asset_existing("Table", "3.1"))
        # Figure 4.2 exists in dataset
        self.assertTrue(is_target_asset_existing("Figure", "4.2"))

    def test_process_vision_element_early_exit(self):
        # Create a mock RunContext with deps
        mock_ctx = MagicMock()
        mock_ctx.deps.vision_element_processed = False
        mock_ctx.deps.user_query = "give overview of Table 6.1"

        res = process_vision_element(mock_ctx, visual_asset_path="tables/Table 6.1.png", extraction_instructions="extract data")
        self.assertIn("NON_EXISTENT_ASSET_ERROR", res)
        self.assertIn("Table 6.1 does not exist in the document dataset", res)


if __name__ == "__main__":
    unittest.main()
