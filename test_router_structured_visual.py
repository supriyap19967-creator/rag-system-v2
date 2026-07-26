import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.main import (
    _chunk_matches_locked_entity,
    _execute_single_query,
    _filter_visual_documents_for_query,
    _visual_results_from_documents,
    bind_image_paths_to_chunks,
    extract_hard_entities,
    hard_entity_strict_label_variants,
)
from app.retriever import RetrievalResult
from app.router_agent import route_query
from app.self_query import build_self_query_hints
from app.structured_query import StructuredConstraint, StructuredQueryResult, StructuredLookup


INDIA_GDP_2022 = Document(
    page_content="In 2022, GDP (current US$) for India (IND) was 3346107287730.93.",
    metadata={
        "source": "Data/csv/GDP1.csv",
        "source_files": "GDP1.csv",
        "source_type": "csv",
        "retrieval_source": "pandas_structured",
        "dataset_type": "NY.GDP.MKTP.CD",
        "country_name": "India",
        "country_iso3": "IND",
        "indicator": "GDP (current US$)",
        "metric_family": "gdp",
        "year": "2022",
        "value": "3346107287730.93",
    },
)

INDIA_CO2_2022 = Document(
    page_content="In 2022, Carbon dioxide (CO2) emissions for India (IND) was 1.96929618962877.",
    metadata={
        "source": "Data/csv/CO21.csv",
        "source_files": "CO21.csv",
        "source_type": "csv",
        "retrieval_source": "pandas_structured",
        "dataset_type": "EN.GHG.CO2.PC.CE.AR5",
        "country_name": "India",
        "country_iso3": "IND",
        "indicator": "Carbon dioxide (CO2) emissions",
        "metric_family": "co2",
        "year": "2022",
        "value": "1.96929618962877",
    },
)

PDF_GROWTH = Document(
    page_content=(
        "The report says economic growth depends on productivity gains, investment, "
        "and stronger institutions."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 10,
    },
)

VISUAL_DOC = Document(
    page_content=(
        "Chart extracted from Vehicle Standards.pdf, page 4. Nearby PDF text: "
        "vehicle emissions standards show a downward trend in allowable emissions."
    ),
    metadata={
        "source": "Data/Pdf/Vehicle Standards.pdf",
        "source_files": "Vehicle Standards.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "chart",
        "topic": "emissions",
        "page": 4,
        "image_path": "Data/extracted_visuals/vehicle-standards-page-4-image-1.png",
        "caption": "Figure 4.6. Vehicle emissions standards.",
        "caption_title": "Vehicle emissions standards.",
        "nearby_text": "Figure 4.6. Vehicle emissions standards show a downward trend.",
        "generated_description": "Vehicle emissions standards chart showing a downward trend.",
    },
)

GENERIC_VISUAL_DOC = Document(
    page_content="Figure extracted from World Development Report, page 1: Environmental standards for development.",
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "figure",
        "topic": "standards",
        "page": 1,
        "image_path": "Data/extracted_visuals/decorative.png",
        "caption": "Figure 1.1. Environmental standards for development.",
        "caption_title": "Environmental standards for development.",
        "nearby_text": "Environmental standards for development.",
    },
)

QUALITY_INFRA_VISUAL_DOC = Document(
    page_content="Figure 3.1 from World Development Report 2025, page 153: Elements of a quality infrastructure system.",
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "figure",
        "topic": "standards",
        "page": 153,
        "image_path": "Data/extracted_visuals/quality-infrastructure.png",
        "caption": "Figure 3.1. Elements of a quality infrastructure system.",
        "caption_title": "Elements of a quality infrastructure system.",
        "nearby_text": "Figure 3.1. Elements of a quality infrastructure system.",
        "extraction_method": "page_crop",
    },
)

TABLE_STANDARDS_DOC = Document(
    page_content="Table 2.2 from World Development Report 2025, page 113: Standards for development and transaction costs.",
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "table",
        "topic": "standards",
        "page": 113,
        "image_path": "Data/extracted_visuals/standards-development-table.png",
        "caption": "Table 2.2. Standards for development and transaction costs.",
        "caption_title": "Standards for development and transaction costs.",
        "nearby_text": "Table 2.2. Standards for development and transaction costs.",
        "extraction_method": "page_crop",
    },
)

FIRMS_LOWER_INCOME_VISUAL_DOC = Document(
    page_content=(
        "[CONTEXT BEFORE]: Standards affect firm competitiveness. | "
        "[VISUAL DATA]: Figure 4.2 Firms in lower-income countries gain proportionately more sales "
        "from adopting voluntary international standards than do firms in more developed countries. | "
        "[CONTEXT AFTER]: The figure compares sales gains across country income groups."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "figure",
        "topic": "firms, lower-income, countries, sales, standards",
        "page": 208,
        "source_page": 208,
        "image_path": "assets/extracted_images/page208_figure1.png",
        "caption": "",
        "original_text": (
            "Figure 4.2 Firms in lower-income countries gain proportionately more sales "
            "from adopting voluntary international standards than do firms in more developed countries."
        ),
        "previous_text": "Standards affect firm competitiveness and help firms improve quality and market access.",
        "next_text": "The figure compares sales gains across country income groups after firms adopt voluntary standards.",
    },
)

WEAK_CHART_VISUAL_DOC = Document(
    page_content="[VISUAL DATA]: Figure 9.9 A weakly extracted chart about standards.",
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "chart",
        "topic": "standards",
        "page": 99,
        "source_page": 99,
        "image_path": "assets/extracted_images/weak.png",
        "caption": "Figure 9.9: A weakly extracted chart about standards.",
        "crop_quality": "chart_expanded_low_quality",
    },
)

TABLE_MARKDOWN_VISUAL_DOC = Document(
    page_content=(
        "[VISUAL DATA]: Table 4.2: Examples of certification costs.\n"
        "| Country | Cost | Standard |\n|---|---:|---|\n| Kenya | 100 | ISO 14001 |\n| India | 80 | ISO 14001 |"
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "table",
        "topic": "certification costs standards",
        "page": 208,
        "source_page": 208,
        "image_path": "assets/extracted_images/page208_table4.png",
        "caption": "Table 4.2: Examples of certification costs for firms in selected markets.",
        "visual_data": "| Country | Cost | Standard |\n|---|---:|---|\n| Kenya | 100 | ISO 14001 |\n| India | 80 | ISO 14001 |",
        "nearby_text": "The report discusses certification costs firms face when adopting standards.",
    },
)

DIAGRAM_VISUAL_DOC = Document(
    page_content="[VISUAL DATA]: Figure 3.1 Elements of a quality infrastructure system connect standards, metrology, accreditation, and conformity assessment.",
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "diagram",
        "topic": "quality infrastructure system standards metrology accreditation",
        "page": 153,
        "source_page": 153,
        "image_path": "assets/extracted_images/quality-infrastructure.png",
        "caption": "Figure 3.1: Elements of a quality infrastructure system.",
        "nearby_text": "Quality infrastructure links standards, metrology, accreditation, and conformity assessment to support reliable markets.",
    },
)

LOWER_INCOME_GENERIC_VISUAL_DOC = Document(
    page_content="Figure 4.1 Lower-income countries use standards in different ways across the economy.",
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_files": "World Development Report 2025.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "figure",
        "topic": "lower-income countries standards",
        "page": 207,
        "source_page": 207,
        "image_path": "assets/extracted_images/page207_figure1.png",
        "caption": "Figure 4.1 Lower-income countries and standards.",
        "original_text": "Figure 4.1 Lower-income countries and standards.",
    },
)

UNRELATED_VISUAL_DOC = Document(
    page_content="Figure 7.1 Tourism receipts by region.",
    metadata={
        "source": "Data/Pdf/Other.pdf",
        "source_files": "Other.pdf",
        "source_type": "pdf",
        "content_type": "visual",
        "visual_type": "figure",
        "topic": "tourism",
        "page": 10,
        "image_path": "assets/extracted_images/tourism.png",
        "caption": "Figure 7.1 Tourism receipts by region.",
        "original_text": "Figure 7.1 Tourism receipts by region.",
    },
)

TABLE_4_1_CHUNK = {
    "content": "Table 4.1 Firms and standards adoption by market segment.",
    "source": "Data/Pdf/World Development Report 2025.pdf",
    "metadata": {
        "content_type": "visual",
        "visual_type": "table",
        "figure_id": "Table 4.1",
        "caption": "Table 4.1 Firms and standards adoption.",
    },
}

FIGURE_4_1_CHUNK = {
    "content": "Figure 4.1 Lower-income countries use standards in different ways across the economy.",
    "source": "Data/Pdf/World Development Report 2025.pdf",
    "metadata": {
        "content_type": "visual",
        "visual_type": "figure",
        "figure_id": "Figure 4.1",
        "caption": "Figure 4.1 Lower-income countries and standards.",
    },
}


class _FakeStructuredEngine:
    def __init__(self, documents):
        self.documents = documents

    def answer(self, question):
        constraints = [
            StructuredConstraint(
                country_name=str(doc.metadata["country_name"]),
                country_iso3=str(doc.metadata["country_iso3"]),
                year=str(doc.metadata["year"]),
                indicator=str(doc.metadata["metric_family"]),
            )
            for doc in self.documents
        ]
        return StructuredQueryResult(
            constraints=constraints,
            lookups=[
                StructuredLookup(constraint=constraint, document=document, source_csv=str(document.metadata["source_files"]))
                for constraint, document in zip(constraints, self.documents)
            ],
            answer_documents=list(self.documents),
            missing_constraints=[],
            engine="pandas",
        )


class RouterStructuredVisualTests(unittest.TestCase):
    def test_table_locked_entity_does_not_match_same_number_figure(self):
        table_entity = extract_hard_entities("Table 4.1")[0]

        self.assertNotIn("4.1", hard_entity_strict_label_variants(table_entity))
        self.assertTrue(_chunk_matches_locked_entity(TABLE_4_1_CHUNK, ["Table 4.1"]))
        self.assertFalse(_chunk_matches_locked_entity(FIGURE_4_1_CHUNK, ["Table 4.1"]))
        self.assertTrue(_chunk_matches_locked_entity(FIGURE_4_1_CHUNK, ["Figure 4.1"]))

    def test_table_request_does_not_reuse_figure_image_path(self):
        table_with_stale_figure_image = {
            **TABLE_4_1_CHUNK,
            "metadata": {
                **TABLE_4_1_CHUNK["metadata"],
                "image_path": "assets/extracted_images/page207_figure1.png",
            },
        }

        image_path = bind_image_paths_to_chunks([table_with_stale_figure_image], ["Table 4.1"])

        self.assertEqual(image_path, "")
        self.assertNotIn("image_path", table_with_stale_figure_image["metadata"])

    def test_router_selects_structured_numeric_route(self):
        decision = route_query("What was India GDP in 2022?")

        self.assertEqual(decision.route, "structured")
        self.assertTrue(decision.use_structured)
        self.assertFalse(decision.use_pdf_retrieval)
        self.assertEqual(decision.constraints[0].country_iso3, "IND")

    def test_numeric_route_does_not_call_vector_retrieval(self):
        with (
            patch("app.main.get_structured_query_engine", return_value=_FakeStructuredEngine([INDIA_GDP_2022])),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("What was India GDP in 2022?", "test", [])

        retrieve.assert_not_called()
        self.assertEqual(result.retrieval_mode, "structured_pandas")
        self.assertIn("India GDP (2022): 3,346,107,287,730.93", result.structured_answer.answer)

    def test_hybrid_route_combines_pandas_and_pdf(self):
        with (
            patch("app.main.get_structured_query_engine", return_value=_FakeStructuredEngine([INDIA_GDP_2022])),
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[PDF_GROWTH], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("What was India GDP in 2022 and explain economic growth?", "test", [])

        self.assertIn("structured_pandas+hybrid", result.retrieval_mode)
        self.assertIn("India GDP (2022): 3,346,107,287,730.93", result.structured_answer.answer)
        self.assertIn("economic growth", result.structured_answer.answer.lower())

    def test_visual_route_returns_visual_document(self):
        with (
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[VISUAL_DOC], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("Show chart about vehicle emissions standards", "test", [])

        self.assertEqual(result.routing["route"], "visual")
        self.assertEqual(result.answer_docs[0].metadata["content_type"], "visual")
        self.assertIn("vehicle emissions standards", result.supporting_evidence.lower())

    def test_visual_relevance_rejects_generic_standards_image(self):
        filtered = _filter_visual_documents_for_query(
            "Show chart about vehicle emissions standards",
            [GENERIC_VISUAL_DOC, VISUAL_DOC],
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].metadata["image_path"], VISUAL_DOC.metadata["image_path"])
        self.assertGreaterEqual(filtered[0].metadata["visual_relevance_score"], 8)

    def test_visual_relevance_handles_required_smoke_queries(self):
        cases = [
            ("Show chart about vehicle emissions standards", VISUAL_DOC),
            ("Show figure about quality infrastructure", QUALITY_INFRA_VISUAL_DOC),
            ("Show table about standards for development", TABLE_STANDARDS_DOC),
        ]

        for query, expected in cases:
            with self.subTest(query=query):
                filtered = _filter_visual_documents_for_query(
                    query,
                    [GENERIC_VISUAL_DOC, VISUAL_DOC, QUALITY_INFRA_VISUAL_DOC, TABLE_STANDARDS_DOC],
                )

                self.assertTrue(filtered)
                self.assertEqual(filtered[0].metadata["image_path"], expected.metadata["image_path"])

    def test_visual_relevance_accepts_captionless_original_text_match(self):
        filtered = _filter_visual_documents_for_query(
            "Show chart about firms in lower-income countries",
            [GENERIC_VISUAL_DOC, LOWER_INCOME_GENERIC_VISUAL_DOC, FIRMS_LOWER_INCOME_VISUAL_DOC],
        )

        self.assertTrue(filtered)
        self.assertEqual(filtered[0].metadata["image_path"], FIRMS_LOWER_INCOME_VISUAL_DOC.metadata["image_path"])
        self.assertIn("Firms in lower-income countries", filtered[0].metadata["caption"])
        self.assertGreaterEqual(filtered[0].metadata["visual_relevance_score"], 8)
        self.assertNotIn(
            LOWER_INCOME_GENERIC_VISUAL_DOC.metadata["image_path"],
            [doc.metadata["image_path"] for doc in filtered],
        )

    def test_visual_route_uses_deterministic_answer_when_llm_unavailable(self):
        with (
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[FIRMS_LOWER_INCOME_VISUAL_DOC], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("Show chart about firms in lower-income countries", "test", [])

        self.assertEqual(result.model_used, "local-visual")
        self.assertTrue(result.answer_docs)
        self.assertIn("Figure 4.2", result.structured_answer.answer)
        self.assertIn("lower-income countries", result.structured_answer.answer)
        self.assertIn("What the visual shows:", result.structured_answer.answer)
        self.assertIn("Key extracted facts:", result.structured_answer.answer)
        self.assertIn("Related paragraph insight:", result.structured_answer.answer)
        self.assertIn("Combined interpretation:", result.structured_answer.answer)
        self.assertIn("Source: World Development Report 2025.pdf, Figure 4.2, page 208.", result.structured_answer.answer)
        self.assertGreaterEqual(result.structured_answer.answer.count("* "), 2)
        self.assertGreaterEqual(result.structured_answer.confidence_score, 0.8)
        self.assertNotIn("values such as 4.2", result.structured_answer.answer)

    def test_weak_visual_answer_includes_limitation_without_numbers(self):
        with (
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[WEAK_CHART_VISUAL_DOC], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("Show chart about standards", "test", [])

        self.assertEqual(
            result.structured_answer.answer,
            "No reliable chart/table/diagram evidence could be extracted for this query from the indexed PDFs.",
        )
        self.assertLessEqual(result.structured_answer.confidence_score, 0.30)
        self.assertEqual(result.answer_docs, [])

    def test_table_visual_answer_extracts_columns_rows_and_comparison(self):
        with (
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[TABLE_MARKDOWN_VISUAL_DOC], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("Show table about certification costs standards", "test", [])

        answer = result.structured_answer.answer
        self.assertIn("Columns identified: Country, Cost, Standard.", answer)
        self.assertIn("Top relevant row: Kenya, 100, ISO 14001.", answer)
        self.assertIn("comparison between Kenya and India", answer)

    def test_diagram_visual_answer_extracts_entities_and_relationship(self):
        with (
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[DIAGRAM_VISUAL_DOC], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("Show diagram about quality infrastructure", "test", [])

        answer = result.structured_answer.answer
        self.assertIn("main entities or stages", answer)
        self.assertIn("relationships among the entities", answer)
        self.assertIn("Related paragraph insight:", answer)

    def test_no_visual_match_returns_clean_message(self):
        with (
            patch("app.main.get_relevant_documents", return_value=RetrievalResult(documents=[UNRELATED_VISUAL_DOC], mode="hybrid")),
            patch("app.main.get_hybrid_llm") as llm,
        ):
            llm.return_value.is_available.return_value = False
            result = _execute_single_query("Show chart about vehicle emissions standards", "test", [])

        self.assertFalse(result.answer_docs)
        self.assertIn("No relevant chart/table found", result.structured_answer.answer)

    def test_visual_results_render_one_or_two_valid_images_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_one = root / "chart-one.png"
            image_two = root / "chart-two.png"
            image_one.write_bytes(b"fake image")
            image_two.write_bytes(b"fake image")
            docs = [
                Document(
                    page_content="Figure 1.1 chart one compares standard adopted and no standard adopted groups with a higher trend on the axis.",
                    metadata={
                        "content_type": "visual",
                        "visual_type": "chart",
                        "source": "report.pdf",
                        "source_files": "report.pdf",
                        "page": 1,
                        "image_path": str(image_one),
                        "caption": "Figure 1.1: Chart one compares standard adopted and no standard adopted groups.",
                        "generated_description": "The chart shows higher values for the standard adopted group and includes an axis trend.",
                        "crop_quality": "chart_complete",
                        "crop_quality_score": 0.8,
                    },
                ),
                Document(
                    page_content="Figure 1.2 chart two compares standard adopted and no standard adopted groups with a higher trend on the axis.",
                    metadata={
                        "content_type": "visual",
                        "visual_type": "chart",
                        "source": "report.pdf",
                        "source_files": "report.pdf",
                        "page": 2,
                        "image_path": str(image_two),
                        "caption": "Figure 1.2: Chart two compares standard adopted and no standard adopted groups.",
                        "generated_description": "The chart shows higher values and distinguishes standard adopted from no standard adopted.",
                        "crop_quality": "chart_complete",
                        "crop_quality_score": 0.8,
                    },
                ),
            ]

            visuals = _visual_results_from_documents(docs)

        self.assertEqual(len(visuals), 2)
        self.assertEqual([visual["page_number"] for visual in visuals], [1, 2])

    def test_visual_results_reject_paragraph_and_incomplete_crops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paragraph_image = root / "paragraph.png"
            incomplete_image = root / "incomplete.png"
            paragraph_image.write_bytes(b"fake image")
            incomplete_image.write_bytes(b"fake image")
            paragraph_doc = Document(
                page_content="Paragraph screenshot about standards.",
                metadata={
                    "content_type": "visual",
                    "visual_type": "paragraph",
                    "source": "report.pdf",
                    "source_files": "report.pdf",
                    "page": 3,
                    "image_path": str(paragraph_image),
                    "caption": "Paragraph text block.",
                    "crop_quality_score": 0.7,
                },
            )
            incomplete_chart = Document(
                page_content="Figure 2.1 incomplete chart.",
                metadata={
                    "content_type": "visual",
                    "visual_type": "chart",
                    "source": "report.pdf",
                    "source_files": "report.pdf",
                    "page": 4,
                    "image_path": str(incomplete_image),
                    "caption": "Figure 2.1: Incomplete chart.",
                    "crop_quality": "chart_expanded_low_quality",
                    "crop_quality_score": 0.2,
                    "crop_rejected_reason": "missing_axis_line;missing_plotted_marks",
                },
            )

            visuals = _visual_results_from_documents([paragraph_doc, incomplete_chart])

        self.assertEqual(visuals, [])

    def test_visual_results_reject_generic_figure_under_strict_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "figure.png"
            image_path.write_bytes(b"fake image")
            generic_figure = Document(
                page_content="Figure 3.1 Elements of a quality infrastructure system connect standards and metrology.",
                metadata={
                    "content_type": "visual",
                    "visual_type": "figure",
                    "source": "report.pdf",
                    "source_files": "report.pdf",
                    "page": 3,
                    "image_path": str(image_path),
                    "caption": "Figure 3.1: Elements of a quality infrastructure system.",
                    "crop_quality": "figure_layout_region_accepted",
                    "crop_quality_score": 0.8,
                },
            )

            visuals = _visual_results_from_documents([generic_figure])

        self.assertEqual(visuals, [])

    def test_self_query_filter_generates_pdf_metadata_hints(self):
        result = build_self_query_hints("What does the report say about India emissions in 2022?")

        self.assertTrue(result.applied)
        self.assertEqual(result.hints.source_type, "pdf")
        self.assertEqual(result.hints.country_iso3, "IND")
        self.assertEqual(result.hints.year, "2022")
        self.assertEqual(result.hints.indicator_family, "co2")


if __name__ == "__main__":
    unittest.main()
