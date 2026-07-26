import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.pdf_visual_extraction import LayoutElement, _clean_caption_text, _trim_visual_crop, extract_pdf_visual_documents


class PdfVisualExtractionTests(unittest.TestCase):
    def test_indexes_unstructured_image_with_caption_and_local_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_dir = root / "pdfs"
            output_dir = root / "assets" / "extracted_images"
            source_image = root / "raw-chart.png"
            pdf_dir.mkdir()
            pdf_path = pdf_dir / "report.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            image = Image.new("RGB", (300, 180), "white")
            pixels = image.load()
            for x in range(45, 270):
                pixels[x, 135] = (20, 20, 20)
            for y in range(35, 140):
                pixels[45, y] = (20, 20, 20)
            for x, y in ((70, 95), (130, 80), (190, 70), (245, 55)):
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        pixels[x + dx, y + dy] = (0, 90, 160)
            image.save(source_image)

            elements = [
                LayoutElement(
                    text="Vehicle policy context before the visual discusses emissions rules.",
                    category="NarrativeText",
                    page_number=7,
                ),
                LayoutElement(
                    text="Figure 4.6 Vehicle emissions standards by region.",
                    category="FigureCaption",
                    page_number=7,
                ),
                LayoutElement(
                    text="",
                    category="Image",
                    page_number=7,
                    image_path=str(source_image),
                ),
                LayoutElement(
                    text="After the figure, the report explains that tighter standards reduce emissions.",
                    category="NarrativeText",
                    page_number=7,
                ),
            ]

            with patch("app.pdf_visual_extraction._partition_pdf", return_value=elements):
                with patch("app.pdf_visual_extraction._caption_image_with_gemini", return_value="The chart compares vehicle emissions standards across regions."):
                    docs = extract_pdf_visual_documents(pdf_dir, output_dir)

            self.assertEqual(len(docs), 1)
            doc = docs[0]
            self.assertEqual(doc.metadata["content_type"], "visual")
            self.assertEqual(doc.metadata["element_type"], "image")
            self.assertEqual(doc.metadata["source_page"], 7)
            self.assertIn("Vehicle emissions standards", doc.metadata["caption"])
            self.assertIn("vehicle emissions standards", doc.page_content.lower())
            self.assertTrue(doc.metadata["is_multimodal"])
            self.assertIn("[CONTEXT BEFORE]:", doc.page_content)
            self.assertIn("[VISUAL DATA]:", doc.page_content)
            self.assertIn("[CONTEXT AFTER]:", doc.page_content)
            self.assertIn("Vehicle policy context before", doc.metadata["previous_text"])
            self.assertIn("tighter standards reduce emissions", doc.metadata["next_text"])
            self.assertTrue(Path(doc.metadata["image_local_path"]).exists())

    def test_indexes_table_with_markdown_ready_caption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_dir = root / "pdfs"
            output_dir = root / "assets" / "extracted_images"
            pdf_dir.mkdir()
            (pdf_dir / "report.pdf").write_bytes(b"%PDF-1.4\n")

            elements = [
                LayoutElement(
                    text="Table 2.1 Standards for development by sector.",
                    category="Table",
                    page_number=3,
                    html="<table><tr><td>Sector</td><td>Standard</td></tr></table>",
                ),
            ]

            with patch("app.pdf_visual_extraction._partition_pdf", return_value=elements):
                with patch("app.pdf_visual_extraction._caption_table_with_gemini", return_value="| Sector | Standard |\n|---|---|\n| Energy | Safety |"):
                    docs = extract_pdf_visual_documents(pdf_dir, output_dir)

            self.assertEqual(len(docs), 1)
            doc = docs[0]
            self.assertEqual(doc.metadata["element_type"], "table")
            self.assertEqual(doc.metadata["visual_type"], "table")
            self.assertEqual(doc.metadata["image_local_path"], "")
            self.assertIn("| Sector | Standard |", doc.metadata["generated_description"])
            self.assertIn("<table>", doc.metadata["visual_data"])
            self.assertIn("[VISUAL DATA]:", doc.page_content)

    def test_skips_uncaptioned_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_dir = root / "pdfs"
            output_dir = root / "assets" / "extracted_images"
            source_image = root / "decorative.png"
            pdf_dir.mkdir()
            (pdf_dir / "report.pdf").write_bytes(b"%PDF-1.4\n")
            Image.new("RGB", (300, 180), "white").save(source_image)

            elements = [
                LayoutElement(text="", category="Image", page_number=1, image_path=str(source_image)),
            ]

            with patch("app.pdf_visual_extraction._partition_pdf", return_value=elements):
                docs = extract_pdf_visual_documents(pdf_dir, output_dir)

            self.assertEqual(docs, [])

    def test_cleans_noisy_duplicate_caption(self):
        caption = _clean_caption_text(
            "Figure Figure 4.2 Firms in lower-income countries Limited access to credit, "
            "managerial know-how gain proportionately more sales from adopting voluntary international standards "
            "than do firms in more developed countries"
        )

        self.assertEqual(
            caption,
            "Figure 4.2: Firms in lower-income countries gain proportionately more sales from adopting voluntary international standards than firms in more developed countries.",
        )

    def test_trims_right_side_text_heavy_crop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_dir = root / "pdfs"
            output_dir = root / "assets" / "extracted_images"
            source_image = root / "wide-chart.png"
            pdf_dir.mkdir()
            (pdf_dir / "report.pdf").write_bytes(b"%PDF-1.4\n")

            image = Image.new("RGB", (800, 320), "white")
            pixels = image.load()
            for x in range(40, 390, 50):
                for y in range(50, 240):
                    pixels[x, y] = (0, 0, 0)
            for y in range(60, 250, 30):
                for x in range(60, 380):
                    pixels[x, y] = (0, 0, 0)
            for y in range(40, 270, 18):
                for x in range(470, 760):
                    pixels[x, y] = (30, 30, 30)
            image.save(source_image)

            elements = [
                LayoutElement(
                    text="Figure 4.2 Firms in lower-income countries gain proportionately more benefits.",
                    category="FigureCaption",
                    page_number=208,
                ),
                LayoutElement(text="", category="Image", page_number=208, image_path=str(source_image)),
            ]

            with patch("app.pdf_visual_extraction._partition_pdf", return_value=elements):
                with patch("app.pdf_visual_extraction._caption_image_with_gemini", return_value="Figure 4.2 shows sales gains from standards."):
                    docs = extract_pdf_visual_documents(pdf_dir, output_dir)

            self.assertEqual(len(docs), 1)
            with Image.open(docs[0].metadata["image_local_path"]) as final_image:
                self.assertLess(final_image.size[0], 760)
            self.assertTrue(Path(docs[0].metadata["raw_image_path"]).exists())

    def test_chart_layout_crop_preserves_full_plot_width(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "figure-4-2.png"
            image = Image.new("RGB", (760, 520), "white")
            pixels = image.load()
            for x in range(120, 700):
                pixels[x, 260] = (40, 40, 40)
            for y in range(120, 360):
                pixels[120, y] = (40, 40, 40)
            for x in range(140, 680, 120):
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        pixels[x + dx, 180 + (x % 4) + dy] = (0, 80, 140)
                        pixels[x + dx, 280 + dy] = (0, 120, 180)
            for y in range(390, 470, 16):
                for x in range(30, 720):
                    if x % 9 in (0, 1):
                        pixels[x, y] = (50, 50, 50)
            image.save(image_path)

            _trim_visual_crop(image_path, trim_right_text=False)

            with Image.open(image_path) as final_image:
                self.assertGreater(final_image.size[0], 650)

    def test_uses_pymupdf_fallback_when_unstructured_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_dir = root / "pdfs"
            output_dir = root / "assets" / "extracted_images"
            pdf_dir.mkdir()
            pdf_path = pdf_dir / "report.pdf"

            import fitz

            doc = fitz.open()
            page = doc.new_page(width=400, height=500)
            page.insert_text((40, 300), "Figure 4.2 Firms in lower-income countries gain more sales from standards.")
            page.draw_rect(fitz.Rect(45, 100, 220, 260))
            doc.save(str(pdf_path))
            doc.close()

            from app.pdf_visual_extraction import _fallback_pymupdf_visual_elements

            elements = _fallback_pymupdf_visual_elements(pdf_path, output_dir, start_page=1, end_page=1)

            self.assertGreaterEqual(len(elements), 2)
            self.assertEqual(elements[0].category, "FigureCaption")
            self.assertIn("Figure 4.2", elements[0].text)
            self.assertTrue(Path(elements[1].image_path).exists())
            self.assertEqual(elements[1].extraction_method, "pymupdf_page_crop")
            self.assertTrue(elements[1].crop_quality)


if __name__ == "__main__":
    unittest.main()
