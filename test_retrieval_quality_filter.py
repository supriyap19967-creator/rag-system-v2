import unittest

from langchain_core.documents import Document

from app.retriever import _filter_retrieved_documents


def pdf_chunk(text: str, rank: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_type": "pdf",
            "source": "World Development Report 2025.pdf",
            "rerank_score": 1.0 / rank,
        },
    )


class RetrievalQualityFilterTest(unittest.TestCase):
    def test_removes_noisy_pdf_chunks_and_keeps_meaningful_text(self):
        noisy_url = pdf_chunk("https://reproducibility.worldbank.org and related replication files", 1)
        noisy_header = pdf_chunk("World Development Report 2025 154", 2)
        noisy_contents = pdf_chunk("Contents xv Foreword xvii Acknowledgments xix", 3)
        useful = pdf_chunk(
            (
                "Standards support development by diffusing good practices across firms and governments. "
                "They can improve efficiency and quality, helping producers participate in trade. "
                "When standards are credible, they can also support growth by reducing uncertainty."
            ),
            4,
        )

        filtered = _filter_retrieved_documents(
            [noisy_url, noisy_header, noisy_contents, useful],
            top_k=3,
        )

        self.assertEqual([document.page_content for document in filtered], [useful.page_content])
        self.assertEqual(filtered[0].metadata["retrieval_quality_status"], "kept_clean")
        self.assertGreaterEqual(filtered[0].metadata["retrieval_quality_score"], 2)

    def test_preserves_csv_chunks_without_pdf_quality_rules(self):
        csv_document = Document(
            page_content="In 2022, GDP (current US$) for India (IND) was 3385090000000.",
            metadata={"source_type": "csv", "source": "GDP.csv"},
        )

        filtered = _filter_retrieved_documents([csv_document], top_k=5)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].page_content, csv_document.page_content)
        self.assertEqual(filtered[0].metadata["retrieval_quality_status"], "kept_non_pdf")


if __name__ == "__main__":
    unittest.main()
