import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from streamlit_ui.StreamlitApp import (
    swap_and_clean_row,
    sanitize_axis_cross_blending_text,
    parse_markdown_table_to_dicts,
    ChartTableRow,
)


class TestVisualAxisFix(unittest.TestCase):

    def test_swap_and_clean_row_axis_inversion(self):
        # Case 1: Category has Y-axis metric (0.6) and TargetValue has X-axis year (2014)
        s, c, v = swap_and_clean_row(series="Data Point", category="0.6", target_val=2014)
        self.assertEqual(c, "2014")
        self.assertEqual(v, "0.6")

    def test_swap_and_clean_row_normal(self):
        # Case 2: Normal orientation (Category is Year 2014, TargetValue is metric 0.6)
        s, c, v = swap_and_clean_row(series="Data Point", category="2014", target_val=0.6)
        self.assertEqual(c, "2014")
        self.assertEqual(v, 0.6)

    def test_sanitize_axis_cross_blending_text(self):
        # Case 1: "showing from 0.6 to 2014"
        text1 = "In Figure 4.2, data was showing from 0.6 to 2014 across countries."
        clean1 = sanitize_axis_cross_blending_text(text1)
        self.assertNotIn("from 0.6 to 2014", clean1)
        self.assertIn("0.6", clean1)
        self.assertIn("2014", clean1)

        # Case 2: "range of 0.6 - 2014"
        text2 = "Values have a range of 0.6 - 2014 in the panel."
        clean2 = sanitize_axis_cross_blending_text(text2)
        self.assertNotIn("range of 0.6 - 2014", clean2)

    def test_parse_markdown_table_headers(self):
        md = """| Panel | Series Name | X-Axis Value | Y-Axis Value |
|---|---|---|---|
| Panel A | Standard adopted | 2014 | 0.6 |"""
        rows = parse_markdown_table_to_dicts(md)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["Series"], "Standard adopted")
        self.assertEqual(row["Category"], "2014")
        self.assertEqual(row["TargetValue"], 0.6)

    def test_chart_table_row_validation(self):
        # Test that model_validator automatically fixes inverted keys
        row = ChartTableRow(Series="Data Point", Category="0.75", TargetValue=2020)
        self.assertEqual(row.Category, "2020")
        self.assertEqual(row.TargetValue, "0.75")


if __name__ == "__main__":
    unittest.main()
