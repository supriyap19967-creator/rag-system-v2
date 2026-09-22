import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.metrics_taxonomy import contains_synthetic_or_dummy_data
from rag_invariants import RAGInvariantsValidator

class TestArchitecturalSolutions(unittest.TestCase):

    def test_layer14_valid_chart_number_not_dummy(self):
        # Valid GDP / chart data with number 20000 for a real entity (Region 2)
        real_chart_data = [
            {"Series": "Region 1", "Category": "GDP per capita", "TargetValue": 10000},
            {"Series": "Region 2", "Category": "GDP per capita", "TargetValue": 20000},
        ]
        has_dummy, explanation = contains_synthetic_or_dummy_data(real_chart_data)
        self.assertFalse(has_dummy, f"Expected valid chart data to pass, but got: {explanation}")

    def test_layer14_synthetic_placeholder_entity_flagged(self):
        # Synthetic data using Country A / Series 1 placeholder
        dummy_chart_data = [
            {"Series": "Country A", "Category": "GDP per capita", "TargetValue": 10000},
        ]
        has_dummy, explanation = contains_synthetic_or_dummy_data(dummy_chart_data)
        self.assertTrue(has_dummy)
        self.assertIn("synthetic placeholder entity", explanation)

    def test_layer06_supabase_url_bypasses_local_disk_check(self):
        invariants = RAGInvariantsValidator()
        supabase_url = "https://furnlyvcinfdoctxtkiu.supabase.co/storage/v1/object/public/rag-assets/extracted_charts/page_208_Figure_4.2.png"
        count = invariants.validate_asset_paths([supabase_url])
        self.assertEqual(count, 1)

if __name__ == "__main__":
    unittest.main()
