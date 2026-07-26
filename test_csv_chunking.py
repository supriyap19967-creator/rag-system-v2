from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ingestion.csv_chunking import detect_csv_header, parse_csv_file


def _write(base: Path, name: str, content: str) -> Path:
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class CsvChunkingTests(unittest.TestCase):
    def test_world_bank_header_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "wb.csv",
                '"Data Source","World Development Indicators"\n'
                '"Last Updated Date","2026-04-08"\n'
                "\n"
                '"Country Name","Country Code","Indicator Name","Indicator Code","2020","2021"\n'
                '"India","IND","GDP (current US$)","NY.GDP.MKTP.CD","10","11"\n',
            )
            header_index, headers, csv_kind = detect_csv_header(path)
            self.assertEqual(header_index, 3)
            self.assertEqual(headers[:4], ["Country Name", "Country Code", "Indicator Name", "Indicator Code"])
            self.assertEqual(csv_kind, "world_bank_wide")

    def test_country_row_chunk_creation_and_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "wb.csv",
                '"Data Source","World Development Indicators"\n'
                '"Last Updated Date","2026-04-08"\n'
                "\n"
                '"Country Name","Country Code","Indicator Name","Indicator Code","2020","2021","2022","2023"\n'
                '"India","IND","GDP (current US$)","NY.GDP.MKTP.CD","10","11","","13"\n',
            )
            parsed = parse_csv_file(path)
            summary = next(block for block in parsed.blocks if block.metadata["entity_type"] == "csv_timeseries")
            range_chunk = next(block for block in parsed.blocks if block.metadata["entity_type"] == "csv_timeseries_range")
            self.assertEqual(summary.metadata["country_code"], "IND")
            self.assertEqual(summary.metadata["indicator_code"], "NY.GDP.MKTP.CD")
            self.assertEqual(summary.metadata["available_years"], [2020, 2021, 2023])
            self.assertEqual(summary.metadata["missing_years"], [2022])
            self.assertIn("India (IND)", summary.text)
            self.assertEqual(range_chunk.metadata["values_by_year"]["2020"], 10)
            self.assertEqual(range_chunk.metadata["values_by_year"]["2021"], 11)
            self.assertEqual(range_chunk.metadata["values_by_year"]["2023"], 13)

    def test_missing_values_are_not_converted_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "wb.csv",
                '"Country Name","Country Code","Indicator Name","Indicator Code","2020","2021"\n'
                '"India","IND","GDP (current US$)","NY.GDP.MKTP.CD","..",""\n',
            )
            parsed = parse_csv_file(path)
            summary = next(block for block in parsed.blocks if block.metadata["entity_type"] == "csv_timeseries")
            self.assertEqual(summary.metadata["available_years"], [])
            self.assertEqual(summary.metadata["missing_years"], [2020, 2021])
            self.assertIn("missing", summary.text.lower())

    def test_country_metadata_chunk_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "country.csv",
                '"Country Code","Region","IncomeGroup","SpecialNotes","TableName"\n'
                '"IND","South Asia","Lower middle income","","India"\n',
            )
            parsed = parse_csv_file(path)
            block = parsed.blocks[0]
            self.assertEqual(parsed.csv_kind, "country_metadata")
            self.assertEqual(block.metadata["entity_type"], "country_metadata")
            self.assertEqual(block.metadata["country_code"], "IND")
            self.assertIn("South Asia", block.text)

    def test_indicator_metadata_chunk_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "indicator.csv",
                '"INDICATOR_CODE","INDICATOR_NAME","SOURCE_NOTE","SOURCE_ORGANIZATION"\n'
                '"NY.GDP.MKTP.CD","GDP (current US$)","A note","World Bank"\n',
            )
            parsed = parse_csv_file(path)
            block = parsed.blocks[0]
            self.assertEqual(parsed.csv_kind, "indicator_metadata")
            self.assertEqual(block.metadata["entity_type"], "indicator_metadata")
            self.assertEqual(block.metadata["indicator_code"], "NY.GDP.MKTP.CD")
            self.assertIn("A note", block.text)

    def test_extracted_table_entity_id_and_row_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "assets/extracted_tables/page_208_Table_4.2.csv",
                '"COUNTRY","VALUE"\n'
                '"India","42"\n'
                '"Brazil","30"\n',
            )
            parsed = parse_csv_file(path)
            self.assertEqual(parsed.csv_kind, "extracted_table_csv")
            summary = next(block for block in parsed.blocks if block.metadata["chunk_id"].endswith("::summary"))
            row_chunk = next(block for block in parsed.blocks if "::row::1" in block.metadata["chunk_id"])
            self.assertEqual(summary.metadata["entity_id"], "Table 4.2")
            self.assertEqual(summary.metadata["page_no"], 208)
            self.assertEqual(row_chunk.metadata["entity_type"], "table")
            self.assertIn("COUNTRY = India", row_chunk.text)

    def test_chunk_ids_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write(
                Path(tmp_dir),
                "wb.csv",
                '"Country Name","Country Code","Indicator Name","Indicator Code","2020"\n'
                '"India","IND","GDP (current US$)","NY.GDP.MKTP.CD","10"\n',
            )
            first = [block.metadata["chunk_id"] for block in parse_csv_file(path).blocks]
            second = [block.metadata["chunk_id"] for block in parse_csv_file(path).blocks]
            self.assertEqual(first, second)

    def test_metadata_exclusivity_for_csv_and_table_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            world_bank_path = _write(
                base,
                "wb.csv",
                '"Country Name","Country Code","Indicator Name","Indicator Code","2020"\n'
                '"India","IND","GDP (current US$)","NY.GDP.MKTP.CD","10"\n',
            )
            table_path = _write(
                base,
                "assets/extracted_tables/page_208_Table_4.2.csv",
                '"COUNTRY","VALUE"\n'
                '"India","42"\n',
            )
            csv_block = parse_csv_file(world_bank_path).blocks[0]
            table_block = parse_csv_file(table_path).blocks[0]
            self.assertFalse(csv_block.metadata["contains_table"])
            self.assertEqual(csv_block.metadata["figure_image_path"], "")
            self.assertEqual(csv_block.metadata["chart_image_path"], "")
            self.assertTrue(table_block.metadata["contains_table"])
            self.assertTrue(table_block.metadata["table_csv_path"].endswith("page_208_Table_4.2.csv"))
            self.assertEqual(table_block.metadata["figure_image_path"], "")
            self.assertEqual(table_block.metadata["chart_image_path"], "")


if __name__ == "__main__":
    unittest.main()
