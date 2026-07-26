import unittest

from embeddings.embed_chunks import ChunkEmbedder
from ingestion.schemas import Chunk


class FakeBgeModel:
    class Settings:
        batch_size = 2
        model_name_or_path = "BAAI/bge-m3"

    settings = Settings()
    backend = "fake"

    def embed_documents(self, texts, batch_size=None):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class BgeM3EmbeddingComponentTests(unittest.TestCase):
    def test_embedder_preserves_multimodal_enrichment_metadata(self):
        chunks = [
            Chunk(
                text=(
                    "Original paragraph.\n\n"
                    "Table summary: this table has 2 data rows.\n\n"
                    "[CHART DESCRIPTION]\nRevenue increased steadily from Q1 to Q4.\n[/CHART DESCRIPTION]"
                ),
                metadata={"source": "annual_report.pdf", "page": 12, "chunk_index": 45},
            )
        ]

        embedded = ChunkEmbedder(model=FakeBgeModel()).embed_chunks(chunks)

        self.assertEqual(len(embedded), 1)
        self.assertTrue(embedded[0].metadata["contains_chart"])
        self.assertTrue(embedded[0].metadata["contains_table"])
        self.assertEqual(embedded[0].metadata["source"], "annual_report.pdf")
        self.assertEqual(embedded[0].metadata["page"], 12)
        self.assertEqual(embedded[0].metadata["embedding_model"], "BAAI/bge-m3")
        self.assertEqual(embedded[0].metadata["embedding_dimension"], 3)


if __name__ == "__main__":
    unittest.main()
