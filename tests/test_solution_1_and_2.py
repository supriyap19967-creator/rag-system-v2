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
        self.assertTrue(url.startswith("https://lurnlyvcinfdoctxtkiu.supabase.co") or url.startswith("https://furnlyvcinfdoctxtkiu.supabase.co"))

    def test_supabase_url_resolution_figure_image(self):
        url = get_supabase_asset_url("Figure_4.2.png")
        self.assertIn("extracted_charts/page_208_Figure_4.2.png", url)
        self.assertTrue(url.startswith("https://lurnlyvcinfdoctxtkiu.supabase.co") or url.startswith("https://furnlyvcinfdoctxtkiu.supabase.co"))

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

    def test_supabase_url_resolution_figure_4_1_no_extension(self):
        url = get_supabase_asset_url("Figure_4.1")
        self.assertIn("extracted_charts/figure_4.1.png", url)

    def test_supabase_url_resolution_figure_4_1_dead_url_no_ext(self):
        url = get_supabase_asset_url("https://furnlyvcinfdoctxtkiu.supabase.co/storage/v1/object/public/rag/extracted_charts/Figure_4.1")
        self.assertIn("extracted_charts/figure_4.1.png", url)

    def test_supabase_asset_base64_generation(self):
        import base64
        import os
        import tempfile
        from app.multimodal_assets import get_supabase_asset_base64

        # 1. Direct Base64 data URL passthrough
        sample_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        self.assertEqual(get_supabase_asset_base64(sample_b64), sample_b64)

        # 2. Local image file conversion to Base64
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
            tmp_path = tmp.name
        try:
            b64_local = get_supabase_asset_base64(tmp_path)
            self.assertIsNotNone(b64_local)
            self.assertTrue(b64_local.startswith("data:image/png;base64,"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 3. Remote / Materialized asset resolution
        b64 = get_supabase_asset_base64("Figure_4.1")
        if b64 is not None:
            self.assertTrue(b64.startswith("data:image/png;base64,"))

    def test_supabase_asset_base64_guards_csv(self):
        from app.multimodal_assets import get_supabase_asset_base64
        b64 = get_supabase_asset_base64("page_154_Table_3.1.csv")
        self.assertIsNone(b64)


if __name__ == "__main__":
    unittest.main()
