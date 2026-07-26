import unittest
import json
from unittest.mock import patch

from langchain_core.documents import Document

from app.main import (
    INSUFFICIENT_DATA_MESSAGE,
    QueryRequest,
    _build_retrieval_queries,
    _generate_guarded_answer,
    _extract_factual_constraints,
    _normalize_user_query,
    rewrite_followup_to_standalone,
    query_rag,
)
from app.retriever import RetrievalResult
from app.schemas import StructuredAnswer
from app.llm import HybridLLM


INDIA_GDP_2022 = Document(
    page_content="In 2022, GDP (current US$) for India (IND) was 3346107287730.93.",
    metadata={
        "source": "Data/csv/GDP1.csv",
        "source_files": "GDP1.csv",
        "source_type": "csv",
        "dataset_type": "NY.GDP.MKTP.CD",
        "country_name": "India",
        "country_iso3": "IND",
        "indicator": "GDP (current US$)",
        "year": "2022",
        "value": "3346107287730.93",
    },
)

INDIA_CO2_2022 = Document(
    page_content="In 2022, Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita) for India (IND) was 1.96929618962877.",
    metadata={
        "source": "Data/csv/CO21.csv",
        "source_files": "CO21.csv",
        "source_type": "csv",
        "dataset_type": "EN.GHG.CO2.PC.CE.AR5",
        "country_name": "India",
        "country_iso3": "IND",
        "indicator": "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
        "year": "2022",
        "value": "1.96929618962877",
    },
)

US_GDP_2022 = Document(
    page_content="In 2022, GDP (current US$) for United States (USA) was 26006893000000.",
    metadata={
        "source": "Data/csv/GDP1.csv",
        "source_files": "GDP1.csv",
        "source_type": "csv",
        "dataset_type": "NY.GDP.MKTP.CD",
        "country_name": "United States",
        "country_iso3": "USA",
        "indicator": "GDP (current US$)",
        "year": "2022",
        "value": "26006893000000",
    },
)

CHINA_GDP_2022 = Document(
    page_content="In 2022, GDP (current US$) for China (CHN) was 17963170052174.1.",
    metadata={
        "source": "Data/csv/GDP1.csv",
        "source_files": "GDP1.csv",
        "source_type": "csv",
        "dataset_type": "NY.GDP.MKTP.CD",
        "country_name": "China",
        "country_iso3": "CHN",
        "indicator": "GDP (current US$)",
        "year": "2022",
        "value": "17963170052174.1",
    },
)

PINECONE_STYLE_INDIA_GDP_2022 = Document(
    page_content="In 2022, GDP (current US$) for India (IND) was 3346107287730.93.",
    metadata={
        "source": "Data/csv/GDP1.csv",
        "source_files": "GDP1.csv",
        "source_type": "csv",
        "dataset_type": "NY.GDP.MKTP.CD",
        "country_name": "India",
        "country_iso3": "IND",
        "indicator": "GDP (current US$)",
        "year": "2022",
        "retrieval_source": "pinecone",
    },
)

KYC_PDF = Document(
    page_content=(
        "The KYC process requires banks to identify customers, verify identity documents, "
        "understand the nature of the customer relationship, and monitor transactions for risk. "
        "These controls help prevent misuse of financial services."
    ),
    metadata={
        "source": "Data/Pdf/KYC-guidance.pdf",
        "source_type": "pdf",
        "page": 2,
    },
)

GROWTH_PDF = Document(
    page_content=(
        "The report says economic growth depends on productivity gains, investment, "
        "and stronger institutions. It also notes that sustained growth requires reforms "
        "that improve market confidence."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 10,
    },
)

STANDARDS_PDF = Document(
    page_content=(
        "Standards for development help diffuse good practices, increase efficiency, "
        "and realize economies of scale by connecting countries through trade and investment. "
        "They also support well-being by improving health and education systems."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 11,
    },
)

REGULATIONS_PDF = Document(
    page_content=(
        "The report says regulations should focus mandatory standards on essential public interests "
        "such as health, safety, environmental protection, and preventing deceptive commercial practices. "
        "It also encourages international regulatory cooperation to reduce fragmented rules."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 12,
    },
)

DEVELOPING_COUNTRIES_STANDARDS_PDF = Document(
    page_content=(
        "Standards help developing countries diffuse good practices and increase efficiency and quality. "
        "They can help firms connect to trade and investment by making products more comparable and trusted. "
        "Stronger quality infrastructure can also support growth, well-being, and risk management."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 13,
    },
)

IMPACT_PDF = Document(
    page_content=(
        "The report says growth improves when productivity, investment, efficiency, and quality rise. "
        "It also notes that climate pressure can constrain development outcomes, so environmental risks "
        "need to be managed alongside economic growth and trade."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 14,
    },
)

REPORT_OVERVIEW_PDF = Document(
    page_content=(
        "World Development Report 2025 Standards for Development. "
        "How standards support development. Standards can be leveraged for diffusing good practices, "
        "increasing efficiency and quality, and helping countries manage risks. "
        "Sources: DieselNet, EU standards lists, and table data 1 2 3 4. "
        "The report also connects standards to growth, trade, investment, well-being, "
        "and stronger quality infrastructure."
    ),
    metadata={
        "source": "Data/Pdf/World Development Report 2025.pdf",
        "source_type": "pdf",
        "page": 1,
    },
)


class FactualQueryUnderstandingTests(unittest.TestCase):
    def unavailable_llm(self):
        class _UnavailableLLM:
            @staticmethod
            def is_available():
                return False

        return _UnavailableLLM()

    def test_generate_guarded_answer_passes_chat_history_to_llm(self):
        captured = {}

        class _MockLLM:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def generate_grounded_answer(**kwargs):
                captured.update(kwargs)
                return {
                    "answer": json.dumps(
                        {
                            "answer": (
                                "India GDP (2022): 3,346,107,287,730.93. "
                                "Future growth can improve when productivity rises, investment expands, "
                                "and reforms strengthen market confidence."
                            ),
                            "confidence_score": 0.88,
                            "source_citations": [
                                {"filename": "GDP1.csv", "page_number": None},
                                {"filename": "Data/Pdf/World Development Report 2025.pdf", "page_number": 10},
                            ],
                        }
                    ),
                    "model_used": "mock-llm",
                }

        deterministic_answer = StructuredAnswer(
            answer=(
                "India GDP (2022): 3,346,107,287,730.93. "
                "The report says economic growth depends on productivity gains, investment, and stronger institutions."
            ),
            confidence_score=0.75,
            source_citations=[],
        )
        chat_history = [
            {"role": "user", "content": "What is GDP?"},
            {"role": "assistant", "content": "GDP is a measure of total output."},
        ]

        with patch("app.main.get_hybrid_llm", return_value=_MockLLM()):
            answer, model_used = _generate_guarded_answer(
                question="How to increase economic growth in future?",
                deterministic_answer=deterministic_answer,
                csv_documents=[INDIA_GDP_2022],
                pdf_documents=[GROWTH_PDF],
                missing_constraints=[],
                requires_factual_validation=True,
                session_id="test",
                chat_history=chat_history,
            )

        self.assertEqual(model_used, "mock-llm")
        self.assertEqual(captured["chat_history"], chat_history)
        self.assertIn("avoid repeating definitions", captured["answer_style"])
        self.assertIn("India GDP (2022): 3,346,107,287,730.93.", answer.answer)

    def test_talkative_prompt_includes_history_and_new_research_data(self):
        captured = {}

        def fake_invoke(self, *, user_prompt, system_prompt, session_id=None):
            captured["user_prompt"] = user_prompt
            captured["system_prompt"] = system_prompt
            captured["session_id"] = session_id
            return {"answer": "{}", "model_used": "mock-llm"}

        with patch.object(HybridLLM, "invoke", fake_invoke):
            result = HybridLLM().get_talkative_answer(
                query="How to increase economic growth in future?",
                context="Productivity, investment, and reforms support growth.",
                history="user: What was India GDP in 2022?\nassistant: India GDP in 2022 was 3.34 trillion US$.",
                instruction="Return JSON only.",
                session_id="test",
            )

        self.assertEqual(result["model_used"], "mock-llm")
        self.assertIn("PREVIOUS CONVERSATION:", captured["user_prompt"])
        self.assertIn("NEW RESEARCH DATA:", captured["user_prompt"])
        self.assertIn("USER'S NEW QUESTION:", captured["user_prompt"])
        self.assertIn("DO NOT repeat", captured["user_prompt"])
        self.assertIn("Return JSON only.", captured["user_prompt"])

    def test_build_retrieval_queries_adds_future_strategy_query(self):
        queries = _build_retrieval_queries(
            "How to increase economic growth in future?",
            constraints=[],
            needs_explanation=True,
        )
        self.assertIn(
            "strategies and future outlook for how to increase economic growth in future",
            queries,
        )

    def assert_has_standard_format(self, response, expected_sources=None):
        self.assertIn("Answer:", response["answer"])
        self.assertIn("Confidence:", response["answer"])
        self.assertIn("Sources:", response["answer"])
        self.assertNotIn("Supporting Evidence", response["answer"])
        if expected_sources is not None:
            for source in expected_sources:
                self.assertIn(source, response["answer"])

    def test_normalizes_possessive_country_phrasing(self):
        self.assertEqual(_normalize_user_query("india's gdp in 2022"), "india gdp in 2022")
        self.assertEqual(_normalize_user_query("indias gdp in 2022"), "india gdp in 2022")
        self.assertEqual(_normalize_user_query("us's gdp for 2022"), "united states gdp for 2022")
        self.assertEqual(_normalize_user_query("U.S. GDP for 2022"), "united states gdp for 2022")

    def test_extracts_constraints_from_natural_variants(self):
        queries = [
            "What was India GDP in 2022?",
            "indias gdp in 2022",
            "india's gdp in 2022",
            "gdp of india 2022",
            "what is india gdp for 2022",
        ]

        for query in queries:
            with self.subTest(query=query):
                constraints = _extract_factual_constraints(query)
                self.assertIsNotNone(constraints)
                self.assertEqual(constraints.country_iso3, "IND")
                self.assertEqual(constraints.indicator, "gdp")
                self.assertEqual(constraints.year, "2022")

    def test_missing_country_is_ambiguous(self):
        self.assertIsNone(_extract_factual_constraints("GDP in 2022?"))

    def test_extracts_us_aliases(self):
        queries = [
            "usa's gdp for 2022",
            "us's gdp for 2022",
            "U.S. GDP for 2022",
            "United States GDP for 2022",
        ]

        for query in queries:
            with self.subTest(query=query):
                constraints = _extract_factual_constraints(query)
                self.assertIsNotNone(constraints)
                self.assertEqual(constraints.country_iso3, "USA")
                self.assertEqual(constraints.indicator, "gdp")
                self.assertEqual(constraints.year, "2022")

    def test_standalone_rewrite_leaves_direct_question_unchanged(self):
        with patch("app.main.get_hybrid_llm", return_value=self.unavailable_llm()):
            rewritten = rewrite_followup_to_standalone(
                "What was India GDP in 2022?",
                [],
            )
        self.assertEqual(rewritten, "What was India GDP in 2022?")

    def test_standalone_rewrite_uses_india_gdp_context(self):
        with patch("app.main.get_hybrid_llm", return_value=self.unavailable_llm()):
            rewritten = rewrite_followup_to_standalone(
                "How to increase economic growth in future?",
                [{"role": "user", "content": "What was India GDP in 2022?"}],
            )
        self.assertEqual(
            rewritten,
            "What strategies or policy recommendations can increase India's GDP and economic growth in the future?",
        )

    def test_standalone_rewrite_uses_india_co2_context(self):
        with patch("app.main.get_hybrid_llm", return_value=self.unavailable_llm()):
            rewritten = rewrite_followup_to_standalone(
                "How can it be reduced?",
                [{"role": "user", "content": "What was India CO2 emission in 2022?"}],
            )
        self.assertEqual(
            rewritten,
            "How can India's CO2 emissions be reduced in the future?",
        )

    def test_standalone_rewrite_uses_compare_context(self):
        with patch("app.main.get_hybrid_llm", return_value=self.unavailable_llm()):
            rewritten = rewrite_followup_to_standalone(
                "What about future growth?",
                [{"role": "user", "content": "Compare India and China GDP in 2022"}],
            )
        self.assertEqual(
            rewritten,
            "What are future growth strategies for India and China based on GDP and economic growth context?",
        )

    def test_standalone_rewrite_leaves_ambiguous_question_without_history_unchanged(self):
        with patch("app.main.get_hybrid_llm", return_value=self.unavailable_llm()):
            rewritten = rewrite_followup_to_standalone(
                "How can it be reduced?",
                [],
            )
        self.assertEqual(rewritten, "How can it be reduced?")

    def test_query_rag_answers_natural_variants_and_rejects_missing_country(self):
        queries = [
            "What was India GDP in 2022?",
            "indias gdp in 2022",
            "india's gdp in 2022",
            "gdp of india 2022",
            "what is india gdp for 2022",
        ]

        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[INDIA_GDP_2022],
                mode="keyword_fallback",
            )

            for query in queries:
                with self.subTest(query=query):
                    response = query_rag(QueryRequest(session_id="test", question=query))
                    self.assert_has_standard_format(response, ["GDP1.csv"])
                    self.assertIn("India GDP (2022):", response["answer"])
                    self.assertEqual(response["sources"], ["GDP1.csv"])

            response = query_rag(QueryRequest(session_id="test", question="GDP in 2022?"))
            self.assert_has_standard_format(response)
            self.assertIn(INSUFFICIENT_DATA_MESSAGE, response["answer"])
            self.assertEqual(response["confidence_score"], 0.1)
            self.assertEqual(response["sources"], [])

    def test_query_rag_answers_us_aliases(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[US_GDP_2022],
                mode="keyword_fallback",
            )

            response = query_rag(QueryRequest(session_id="test", question="us's gdp for 2022"))
            self.assert_has_standard_format(response, ["GDP1.csv"])
            self.assertIn("United States GDP (2022):", response["answer"])
            self.assertEqual(response["sources"], ["GDP1.csv"])

    def test_query_rag_answers_multiple_csv_metrics(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[INDIA_GDP_2022, INDIA_CO2_2022],
                mode="keyword_fallback",
            )

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="What was India GDP in 2022 and CO2 emission in 2022?",
                )
            )
            self.assert_has_standard_format(response, ["GDP1.csv", "CO21.csv"])
            self.assertIn("India GDP (2022):", response["answer"])
            self.assertIn("India CO2 emissions (2022):", response["answer"])
            self.assertEqual(response["sources"], ["CO21.csv", "GDP1.csv"])

    def test_query_rag_formats_pinecone_csv_without_value_metadata(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[PINECONE_STYLE_INDIA_GDP_2022],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="What was India GDP in 2022?"))
            self.assert_has_standard_format(response, ["GDP1.csv"])
            self.assertIn("3,346,107,287,730.93", response["answer"])
            self.assertNotIn("None", response["answer"])

    def test_query_rag_formats_pdf_explanation(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[KYC_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="What is KYC process?"))
            self.assert_has_standard_format(response, ["Data/Pdf/KYC-guidance.pdf"])
            self.assertIn("identify customers", response["answer"])
            self.assertNotIn("source_type", response["answer"])
            self.assertEqual(response["sources"], ["Data/Pdf/KYC-guidance.pdf"])

    def test_query_rag_summarizes_standards_pdf_answer(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[STANDARDS_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="Explain standards for development"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("Based on the retrieved context", response["answer"])
            self.assertIn("increase efficiency", response["answer"])
            self.assertNotIn("source_type", response["answer"])

    def test_query_rag_summarizes_regulations_pdf_answer(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[REPORT_OVERVIEW_PDF, REGULATIONS_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="What does the report say about regulations?"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("The report says", response["answer"])
            self.assertIn("regulations should focus", response["answer"])
            self.assertIn("mandatory standards", response["answer"])
            self.assertIn("regulatory cooperation", response["answer"])
            self.assertNotIn("support development by spreading good practices", response["answer"])
            self.assertNotIn("source_type", response["answer"])

    def test_query_rag_summarizes_developing_country_standards_cleanly(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[DEVELOPING_COUNTRIES_STANDARDS_PDF],
                mode="hybrid",
            )

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="Why are standards important for developing countries?",
                )
            )
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("standards matter for developing countries", response["answer"])
            self.assertIn("trade and investment", response["answer"])
            self.assertLessEqual(response["answer"].count(". "), 3)
            self.assertNotIn("source_type", response["answer"])

    def test_query_rag_adds_pdf_explanation_to_multi_metric_impact_question(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[INDIA_GDP_2022, INDIA_CO2_2022, IMPACT_PDF],
                mode="hybrid",
            )

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="What was India GDP and CO2 emission in 2022 and explain their impact?",
                )
            )
            self.assert_has_standard_format(response, ["GDP1.csv", "CO21.csv", "Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("India GDP (2022):", response["answer"])
            self.assertIn("India CO2 emissions (2022):", response["answer"])
            self.assertIn("GDP reflects economic scale", response["answer"])
            self.assertIn("CO2 emissions point to environmental pressure", response["answer"])
            self.assertIn("Data/Pdf/World Development Report 2025.pdf", response["sources"])

    def test_query_rag_summarizes_generic_standards_report_without_noise(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[REPORT_OVERVIEW_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="tell me about standards report"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("support development", response["answer"])
            self.assertIn("quality infrastructure", response["answer"])
            self.assertNotIn("DieselNet", response["answer"])

    def test_query_rag_composes_hybrid_csv_and_pdf_answer(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[INDIA_GDP_2022, GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="What was India GDP in 2022 and what does the report say about economic growth?",
                )
            )
            self.assert_has_standard_format(response, ["GDP1.csv", "Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("India GDP (2022):", response["answer"])
            self.assertIn("The report says", response["answer"])
            self.assertIn("economic growth depends", response["answer"])
            self.assertEqual(response["sources"], ["Data/Pdf/World Development Report 2025.pdf", "GDP1.csv"])

    def test_query_rag_formats_compare_answer_on_one_line_with_consistent_year(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.side_effect = [
                RetrievalResult(documents=[INDIA_GDP_2022], mode="keyword_fallback"),
                RetrievalResult(documents=[INDIA_GDP_2022], mode="keyword_fallback"),
                RetrievalResult(documents=[CHINA_GDP_2022], mode="keyword_fallback"),
                RetrievalResult(documents=[CHINA_GDP_2022], mode="keyword_fallback"),
            ]

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="Compare India and China GDP in 2022",
                )
            )
            self.assert_has_standard_format(response, ["GDP1.csv"])
            self.assertIn("India GDP (2022):", response["answer"])
            self.assertIn("China GDP (2022):", response["answer"])
            self.assertEqual(response["answer"].count("(2022)"), 2)
            self.assertEqual(response["answer"].count("Answer:"), 1)

    def test_query_rag_synthesizes_open_ended_growth_answer(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="how to achieve economic growth?"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("productivity rises", response["answer"])
            self.assertIn("investment expands", response["answer"])
            self.assertNotIn("depends on productivity gains, investment, and stronger institutions", response["answer"])
            self.assertLessEqual(response["answer"].split("Confidence:")[0].count(". "), 4)

    def test_query_rag_synthesizes_what_do_you_think_question(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="what do you think about growth?"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("productivity rises", response["answer"])
            self.assertIn("steady reforms", response["answer"])
            self.assertNotIn("The report says economic growth depends", response["answer"])

    def test_query_rag_synthesizes_tell_me_growth_question(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="tell me about economic growth"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("productivity rises", response["answer"])
            self.assertIn("investment expands", response["answer"])
            self.assertNotIn("depends on productivity gains, investment, and stronger institutions", response["answer"])

    def test_query_rag_synthesizes_future_growth_question(self):
        with (
            patch("app.main.models_loaded", True),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(QueryRequest(session_id="test", question="how to increase economic growth in future"))
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
            self.assertIn("Future economic growth is more likely", response["answer"])
            self.assertIn("steady reforms", response["answer"])
            self.assertNotIn("The report says economic growth depends", response["answer"])

    def test_query_rag_uses_standalone_rewrite_for_followup(self):
        with (
            patch("app.main.models_loaded", True),
            patch(
                "app.main.fetch_chat_history",
                return_value=[{"role": "user", "content": "What was India GDP in 2022?"}],
            ),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="How to increase economic growth in future?",
                )
            )
            self.assertEqual(
                response["rewritten_query"],
                "what strategies or policy recommendations can increase india gdp and economic growth in the future",
            )
            self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])

    def test_query_rag_followup_synthesis_receives_previous_conversation(self):
        history = [
            {"role": "user", "content": "What was India GDP in 2022?"},
            {"role": "assistant", "content": "India GDP in 2022 was 3,346,107,287,730.93."},
        ]

        captured = {}

        class _MockLLM:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def embed_text(_text):
                return [0.1, 0.2, 0.3]

            @staticmethod
            def rewrite_query(user_input, _chat_history, session_id=None):
                return user_input

            @staticmethod
            def generate_grounded_answer(**kwargs):
                captured.update(kwargs)
                return {
                    "answer": json.dumps(
                        {
                            "answer": (
                                "India GDP (2022): 3,346,107,287,730.93. "
                                "Future growth is more likely when productivity improves, investment remains strong, "
                                "and reforms help sustain market confidence."
                            ),
                            "confidence_score": 0.87,
                            "source_citations": [
                                {"filename": "GDP1.csv", "page_number": None},
                                {"filename": "Data/Pdf/World Development Report 2025.pdf", "page_number": 10},
                            ],
                        }
                    ),
                    "model_used": "mock-llm",
                }

        with (
            patch("app.main.models_loaded", True),
            patch("app.main.fetch_chat_history", return_value=history),
            patch("app.main.get_hybrid_llm", return_value=_MockLLM()),
            patch("app.main.semantic_cache.get", return_value=None),
            patch("app.main.semantic_cache.set"),
            patch("app.main.get_relevant_documents") as retrieve,
            patch("app.main.store_chat_message"),
            patch("app.main.get_total_session_cost", return_value=0.0),
            patch("app.main.get_last_query_cost", return_value={"total": 0.0}),
        ):
            retrieve.return_value = RetrievalResult(
                documents=[INDIA_GDP_2022, GROWTH_PDF],
                mode="hybrid",
            )

            response = query_rag(
                QueryRequest(
                    session_id="test",
                    question="How to increase economic growth in future?",
                )
            )

        self.assert_has_standard_format(response, ["Data/Pdf/World Development Report 2025.pdf"])
        self.assertIn("India GDP (2022): 3,346,107,287,730.93.", response["answer"])
        self.assertIn("Future growth is more likely", response["answer"])
        self.assertNotIn("Answer\nAnswer:", response["answer"])
        self.assertEqual(captured["chat_history"], history)


if __name__ == "__main__":
    unittest.main()
