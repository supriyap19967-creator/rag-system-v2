import unittest

from embeddings.embed_chunks import EmbeddedChunk
from vectordb.ingest_vectors import stable_point_id
from vectordb.metadata_schema import normalize_payload
from vectordb.retrieval_pipeline import ConversationalRetrievalPipeline
from vectordb.search_vectors import SearchResult


class FakeEmbedder:
    def embed_query(self, query):
        self.query = query
        return [1.0, 0.0, 0.0]


class FakeSearcher:
    def search(self, query_vector, top_k=5, filters=None, score_threshold=None):
        self.query_vector = query_vector
        self.filters = filters
        return [
            SearchResult(
                id="point-1",
                text="[CHART DESCRIPTION]\nRevenue rose in Q4.\n[/CHART DESCRIPTION]",
                score=0.91,
                metadata={"source_file": "annual_report.pdf", "contains_chart": True},
                payload={},
            )
        ]


class QdrantComponentTests(unittest.TestCase):
    def test_payload_normalization_preserves_multimodal_fields(self):
        payload = normalize_payload(
            "[CHART DESCRIPTION]\nRevenue rose.\n[/CHART DESCRIPTION]",
            {
                "source": "Data/Pdf/annual_report.pdf",
                "source_type": "pdf",
                "page": "12",
                "chunk_id": "chunk_45",
                "contains_chart": True,
                "contains_table": True,
                "image_path": "chart_5.png",
                "section": "Quarterly Revenue",
            },
        ).to_qdrant_payload()

        self.assertEqual(payload["source_file"], "annual_report.pdf")
        self.assertEqual(payload["page"], 12)
        self.assertTrue(payload["contains_chart"])
        self.assertTrue(payload["contains_table"])
        self.assertEqual(payload["image_reference"], "chart_5.png")

    def test_stable_point_id_is_deterministic_uuid(self):
        chunk = EmbeddedChunk(
            id="chunk_45",
            text="Revenue rose.",
            embedding=[1.0, 0.0, 0.0],
            metadata={"source": "annual_report.pdf", "page": 12},
        )

        self.assertEqual(stable_point_id(chunk), stable_point_id(chunk))
        self.assertEqual(len(stable_point_id(chunk)), 36)

    def test_conversational_retrieval_preserves_results(self):
        embedder = FakeEmbedder()
        searcher = FakeSearcher()
        pipeline = ConversationalRetrievalPipeline(embedder=embedder, searcher=searcher)

        context = pipeline.retrieve(
            "What happened to revenue?",
            conversation_history=[{"role": "user", "content": "Focus on Q4 charts."}],
            filters={"contains_chart": True},
        )

        self.assertIn("Conversation context", context.rewritten_query)
        self.assertEqual(searcher.query_vector, [1.0, 0.0, 0.0])
        self.assertEqual(searcher.filters, {"contains_chart": True})
        self.assertTrue(context.results[0].metadata["contains_chart"])


if __name__ == "__main__":
    unittest.main()
