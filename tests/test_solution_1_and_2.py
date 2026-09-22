import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.multimodal_assets import get_supabase_asset_url

class TestSolution1And2(unittest.TestCase):

    def test_supabase_url_resolution_table_csv(self):
        url = get_supabase_asset_url("page_154_Table_3.1.csv")
        self.assertIn("assets/extracted_tables/page_154_Table_3.1.csv", url)
        self.assertTrue(url.startswith("https://furnlyvcinfdoctxtkiu.supabase.co"))

    def test_supabase_url_resolution_figure_image(self):
        url = get_supabase_asset_url("Figure_4.2.png")
        self.assertIn("extracted_charts/page_208_Figure_4.2.png", url)
        self.assertTrue(url.startswith("https://furnlyvcinfdoctxtkiu.supabase.co"))

    def test_supabase_url_resolution_alias_img_04_02(self):
        url = get_supabase_asset_url("img_04_02")
        self.assertIn("extracted_charts/page_208_Figure_4.2.png", url)

    def test_supabase_url_fallback_csv(self):
        url = get_supabase_asset_url("custom_dataset.csv")
        self.assertIn("assets/extracted_tables/custom_dataset.csv", url)

    def test_supabase_url_fallback_image(self):
        url = get_supabase_asset_url("custom_chart.png")
        self.assertIn("extracted_charts/custom_chart.png", url)

    def test_supabase_url_full_path(self):
        url = get_supabase_asset_url("/mount/src/rag-system-v2/assets/extracted_tables/page_154_Table_3.1.csv")
        self.assertIn("assets/extracted_tables/page_154_Table_3.1.csv", url)

if __name__ == "__main__":
    unittest.main()
