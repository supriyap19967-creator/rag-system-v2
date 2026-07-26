import unittest
import tempfile
from pathlib import Path

from ingestion.chunking import MarkdownChunker
from ingestion.parse_csv import CsvSemanticParser
from ingestion.schemas import EnrichedDocument


class MultimodalIngestionComponentTests(unittest.TestCase):
    def test_csv_parser_emits_structured_blocks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            path.write_text(
                '"Country Name","Country Code","Indicator Name","Indicator Code","2020"\n'
                '"India","IND","GDP (current US$)","NY.GDP.MKTP.CD","10"\n',
                encoding="utf-8",
            )
            document = CsvSemanticParser().parse(path)
            self.assertTrue(document.blocks)
            self.assertEqual(document.blocks[0].metadata["entity_type"], "csv_timeseries")
            self.assertIn("India (IND)", document.blocks[0].text)

    def test_chunker_preserves_chart_description_block(self):
        markdown = """
# Report

This paragraph introduces the report.

[CHART DESCRIPTION]

Sales rose from Q1 to Q4. Q4 had the highest value.

[/CHART DESCRIPTION]
""".strip()
        document = EnrichedDocument(source_path="report.pdf", markdown=markdown)

        chunks = MarkdownChunker(chunk_size=80, chunk_overlap=0).chunk(document)

        chart_chunks = [chunk for chunk in chunks if "[CHART DESCRIPTION]" in chunk.text]
        self.assertEqual(len(chart_chunks), 1)
        self.assertIn("[/CHART DESCRIPTION]", chart_chunks[0].text)


if __name__ == "__main__":
    unittest.main()
