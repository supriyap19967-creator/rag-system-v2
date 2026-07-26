from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion.chunking import MarkdownChunker
from ingestion.pdf_chunking import PdfPageBlock, build_pdf_document, build_visual_blocks
from ingestion.schemas import EnrichedDocument, ExtractedImage, VisionDescription
from streamlit_ui.Streamlitapp import expand_reranked_children_to_parents


def _page_blocks() -> list[PdfPageBlock]:
    return [
        PdfPageBlock(page_no=1, text="World Development Report 2025", bbox=[0, 0, 100, 20], index=1),
        PdfPageBlock(page_no=1, text="Chapter 4: Standards for a Better Economy", bbox=[0, 20, 100, 40], index=2),
        PdfPageBlock(page_no=1, text="4.1 Standards adoption", bbox=[0, 40, 100, 60], index=3),
        PdfPageBlock(page_no=1, text="Firms benefit from adopting standards in multiple ways.", bbox=[0, 60, 100, 80], index=4),
        PdfPageBlock(page_no=1, text="Figure 4.2 Firms in lower-income countries gain proportionately more sales from adopting voluntary international standards.", bbox=[0, 80, 100, 100], index=5),
        PdfPageBlock(page_no=1, text="Source: WDR 2025 team.", bbox=[0, 100, 100, 120], index=6),
        PdfPageBlock(page_no=1, text="The figure compares impacts across country income groups.", bbox=[0, 120, 100, 140], index=7),
    ]


class PdfChunkingPipelineTests(unittest.TestCase):
    @patch("ingestion.pdf_chunking.extract_pdf_page_blocks")
    def test_chapter_and_section_metadata_are_preserved(self, mocked_extract) -> None:
        mocked_extract.return_value = _page_blocks()
        pdf_path = Path("report.pdf")
        markdown = """
# Chapter 4: Standards for a Better Economy

## 4.1 Standards adoption

Firms benefit from adopting standards in multiple ways.

Figure 4.2 Firms in lower-income countries gain proportionately more sales from adopting voluntary international standards.

Source: WDR 2025 team.

The figure compares impacts across country income groups.
""".strip()
        document = build_pdf_document(pdf_path, markdown)
        chunks = MarkdownChunker(chunk_size=1200, chunk_overlap=0).chunk(document)
        text_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("chunk_type") == "section_text_chunk"
            and chunk.metadata.get("chapter_number") == "4"
        ]
        self.assertTrue(text_chunks)
        for chunk in text_chunks:
            self.assertEqual(chunk.metadata["chapter_number"], "4")
            self.assertIn("Chapter 4", chunk.metadata["chapter_title"])
            self.assertEqual(chunk.metadata["section_title"], "4.1 Standards adoption")

    @patch("ingestion.pdf_chunking.extract_pdf_page_blocks")
    def test_visual_heading_becomes_caption_chunk(self, mocked_extract) -> None:
        mocked_extract.return_value = _page_blocks()
        markdown = """
# Chapter 4: Standards for a Better Economy

Figure 4.2 Firms in lower-income countries gain proportionately more sales from adopting voluntary international standards.
""".strip()
        document = build_pdf_document(Path("report.pdf"), markdown)
        visual_candidates = document.metadata["visual_candidates"]
        self.assertEqual(len(visual_candidates), 1)
        self.assertEqual(visual_candidates[0]["entity_id"], "Figure_4.2")
        self.assertIn("Figure 4.2", visual_candidates[0]["caption_text"])

    @patch("ingestion.pdf_chunking.validate_asset_path")
    @patch("ingestion.pdf_chunking.extract_pdf_page_blocks")
    def test_asset_linking_and_exclusivity(self, mocked_extract, mocked_validate_path) -> None:
        mocked_extract.return_value = _page_blocks()
        mocked_validate_path.return_value = type(
            "Validation",
            (),
            {"ok": True, "reason": "verified", "action": "allowed", "path": "ok"},
        )()
        markdown = """
# Chapter 4: Standards for a Better Economy

Figure 4.2 Firms in lower-income countries gain proportionately more sales from adopting voluntary international standards.
""".strip()
        document = build_pdf_document(Path("report.pdf"), markdown)
        visual_candidates = document.metadata["visual_candidates"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "page_1_Figure_4.2.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            image = ExtractedImage(
                image_path=image_path,
                page=1,
                type="figure",
                source_path="report.pdf",
                element_id="img-1",
                coordinates={"bbox": [1, 2, 3, 4]},
                metadata={"entity_id": "Figure_4.2", "source_label": "Figure 4.2 Firms..."},
            )
            description = VisionDescription(
                image_path=image_path,
                description="A bar chart comparing lower-income and higher-income country sales gains.",
                page=1,
                type="chart",
                metadata={"source_label": "Figure 4.2 Firms..."},
            )
            blocks = build_visual_blocks(
                pdf_path=Path("report.pdf"),
                document=document,
                visual_candidates=[type("Candidate", (), candidate)() for candidate in visual_candidates],
                images=[image],
                descriptions=[description],
            )
            asset_chunk = next(block for block in blocks if block.metadata.get("chunk_type") == "visual_asset_chunk")
            self.assertEqual(asset_chunk.metadata["entity_id"], "Figure_4.2")
            self.assertTrue(asset_chunk.metadata["asset_exists"])
            self.assertEqual(asset_chunk.metadata["chart_image_path"], str(image_path.resolve()))
            self.assertEqual(asset_chunk.metadata.get("table_csv_path", ""), "")

    def test_exact_visual_queries_preserve_child_text(self) -> None:
        expanded = expand_reranked_children_to_parents(
            [
                {
                    "id": "child-1",
                    "content": "Figure 4.2 caption text",
                    "metadata": {
                        "parent_id": "parent-1",
                        "parent_text": "A broader page summary mentioning Figure 4.2 and Table 4.1",
                        "preserve_child_text": True,
                    },
                }
            ]
        )
        self.assertEqual(expanded[0]["content"], "Figure 4.2 caption text")
        self.assertIn("Table 4.1", expanded[0]["supporting_parent_text"])

    @patch("ingestion.pdf_chunking.extract_pdf_page_blocks")
    def test_narrative_mention_of_figure_stays_in_text_chunk(self, mocked_extract) -> None:
        mocked_extract.return_value = _page_blocks()
        markdown = """
# Chapter 4: Standards for a Better Economy

## 4.1 Standards adoption

As discussed in Figure 4.2, firms in lower-income countries gain more from standards adoption.

Figure 4.2 Firms in lower-income countries gain proportionately more sales from adopting voluntary international standards.
""".strip()
        document = build_pdf_document(Path("report.pdf"), markdown)
        text_blocks = [block for block in document.blocks if block.metadata.get("chunk_type") == "section_text_chunk"]
        self.assertTrue(any("As discussed in Figure 4.2" in block.text for block in text_blocks))
        self.assertEqual(len(document.metadata["visual_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
