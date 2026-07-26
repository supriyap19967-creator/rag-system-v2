import unittest

from langchain_core.documents import Document

from app.llamaindex_brain import _exact_csv_answer, _exact_csv_documents, build_query
from app.llamaindex_pipeline import DEFAULT_CSV_DIR, _load_csv_nodes


class LlamaIndexChunkingTests(unittest.TestCase):
    def test_csv_chunks_are_row_level_with_exact_value_metadata(self):
        nodes = _load_csv_nodes(DEFAULT_CSV_DIR)
        india_gdp_2022 = [
            node
            for node in nodes
            if node.metadata.get("country_iso3") == "IND"
            and node.metadata.get("year") == "2022"
            and node.metadata.get("metric_family") == "gdp"
        ]

        self.assertTrue(india_gdp_2022)
        node = india_gdp_2022[0]
        self.assertEqual(
            node.page_content,
            "In 2022, GDP (current US$) for India (IND) was 3346107287730.93.",
        )
        self.assertEqual(node.metadata["value"], "3346107287730.93")
        self.assertEqual(node.metadata["country_code"], "IND")

    def test_exact_csv_lookup_does_not_return_nearest_country(self):
        documents = [
            Document(
                page_content="In 2020, GDP (current US$) for India (IND) was 2674851578587.27.",
                metadata={
                    "source_type": "csv",
                    "source_files": "GDP1.csv",
                    "country_name": "India",
                    "country_iso3": "IND",
                    "country_code": "IND",
                    "year": "2020",
                    "indicator": "GDP (current US$)",
                    "metric_family": "gdp",
                    "value": "2674851578587.27",
                },
            )
        ]

        self.assertEqual(_exact_csv_documents("GDP of Atlantis in 2020", documents), [])
        self.assertEqual(_exact_csv_answer("GDP of Atlantis in 2020", []), "")

    def test_query_builder_routes_structured_visual_and_text_queries(self):
        structured = build_query("GDP of India in 2022")
        self.assertEqual(structured.route, "structured_exact")
        self.assertEqual(structured.namespace_order, [])

        visual = build_query("show figure 6.4")
        self.assertEqual(visual.route, "visual_exact")
        self.assertEqual(visual.figure_id, "Figure 6.4")

        semantic = build_query("what does the report say about air pollution standards")
        self.assertEqual(semantic.route, "text_semantic")

        hybrid = build_query("vehicle emissions standards")
        self.assertEqual(hybrid.route, "hybrid_short")


if __name__ == "__main__":
    unittest.main()
