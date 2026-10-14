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

    def test_case_insensitive_figure_and_table_resolution(self):
        # 1. Figure 7.1 in various cases
        for q in ["Figure 7.1", "figure 7.1", "FIGURE 7.1", "Figure_7.1.png", "extracted_charts/Figure_7.1.png"]:
            url = get_supabase_asset_url(q)
            self.assertIn("extracted_charts/figure_7.1.png", url, f"Failed for {q}")

        # 2. Figures 7.2, 4.3, 3.1 in various cases
        for target in ["7.2", "4.3", "3.1"]:
            url_cap = get_supabase_asset_url(f"Figure_{target}.png")
            url_low = get_supabase_asset_url(f"figure {target}")
            self.assertIn(f"extracted_charts/figure_{target}.png", url_cap)
            self.assertIn(f"extracted_charts/figure_{target}.png", url_low)

        # 3. Tables in various cases
        url_t = get_supabase_asset_url("Table 4.1")
        self.assertIn("assets/extracted_tables/page_207_Table_4.1.csv", url_t)

        url_t3 = get_supabase_asset_url("table 3.1")
        self.assertIn("assets/extracted_tables/page_154_Table_3.1.csv", url_t3)

        # 4. Previously broken capitalized HTTP URLs
        broken_url = "https://lurnlyvcinfdoctxtkiu.supabase.co/storage/v1/object/public/rag/extracted_charts/Figure_7.1.png"
        fixed_url = get_supabase_asset_url(broken_url)
        self.assertIn("extracted_charts/figure_7.1.png", fixed_url)

    def test_supabase_asset_base64_guards_csv(self):
        from app.multimodal_assets import get_supabase_asset_base64
        b64 = get_supabase_asset_base64("page_154_Table_3.1.csv")
        self.assertIsNone(b64)

    def test_cloud_manifest_existing_asset_detection(self):
        from streamlit_ui.StreamlitApp import is_target_asset_existing
        # Positive checks (irrespective of case)
        self.assertTrue(is_target_asset_existing("Figure", "7.1"))
        self.assertTrue(is_target_asset_existing("figure", "7.1"))
        self.assertTrue(is_target_asset_existing("FIGURE", "7.1"))
        self.assertTrue(is_target_asset_existing("Figure", "7.2"))
        self.assertTrue(is_target_asset_existing("Figure", "4.3"))
        self.assertTrue(is_target_asset_existing("Figure", "3.1"))
        self.assertTrue(is_target_asset_existing("Table", "3.1"))
        self.assertTrue(is_target_asset_existing("table", "4.1"))

        # Negative checks (non-existent assets)
        self.assertFalse(is_target_asset_existing("Table", "6.1"))
        self.assertFalse(is_target_asset_existing("Figure", "9.9"))


if __name__ == "__main__":
    unittest.main()
