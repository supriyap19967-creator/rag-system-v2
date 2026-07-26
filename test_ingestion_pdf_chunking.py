import unittest

from app.ingestion import _chunk_text


class IngestionPdfChunkingTests(unittest.TestCase):
    def test_removes_noisy_pdf_fragments_and_keeps_paragraphs(self):
        raw_text = """
        World Development Report 2025 154
        Contents
        Chapter 1 ........ 12

        Standards support development by diffusing good practices across firms and governments.
        They increase efficiency and quality, and they help producers participate in trade.

        https://example.com/references
        References
        Journal of Testing, vol. 12, no. 4, pp. 10-22.
        """

        chunks = _chunk_text(raw_text, chunk_size=500, overlap=50)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Standards support development", chunks[0])
        self.assertNotIn("Contents", chunks[0])
        self.assertNotIn("https://example.com", chunks[0])
        self.assertNotIn("Journal of Testing", chunks[0])


if __name__ == "__main__":
    unittest.main()
