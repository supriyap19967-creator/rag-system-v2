import unittest

from app.retriever import RetrievalHints, _metadata_filter_for_hints


class RetrieverMetadataFilterTests(unittest.TestCase):
    def test_builds_strict_metadata_filter_for_structured_csv_queries(self):
        hints = RetrievalHints(
            source_type="csv",
            country_iso3="IND",
            year="2022",
            indicator_family="gdp",
        )

        metadata_filter = _metadata_filter_for_hints(hints, relaxed=False)

        self.assertEqual(
            metadata_filter,
            {
                "$and": [
                    {"source_type": {"$eq": "csv"}},
                    {"country_iso3": {"$eq": "IND"}},
                    {"year": {"$eq": "2022"}},
                    {"metric_family": {"$eq": "gdp"}},
                ]
            },
        )

    def test_relaxed_filter_drops_metric_family_when_needed(self):
        hints = RetrievalHints(
            source_type="csv",
            country_iso3="IND",
            year="2022",
            indicator_family="gdp",
        )

        metadata_filter = _metadata_filter_for_hints(hints, relaxed=True)

        self.assertEqual(
            metadata_filter,
            {
                "$and": [
                    {"source_type": {"$eq": "csv"}},
                    {"country_iso3": {"$eq": "IND"}},
                    {"year": {"$eq": "2022"}},
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
