from ingestion.chunking import MarkdownChunker
from ingestion.schemas import EnrichedDocument
from vectordb.metadata_schema import normalize_payload, qdrant_payload_indexes


def test_markdown_chunker_persists_chapter_hierarchy_metadata() -> None:
    document = EnrichedDocument(
        source_path="report.pdf",
        markdown=(
            "# Chapter 4: Public Finance\n\n"
            "Opening chapter text.\n\n"
            "## 4.1 Budget Systems\n\n"
            "Budget system details.\n\n"
            "### 4.1.1 Procurement Controls\n\n"
            "Procurement control details."
        ),
        metadata={"source_type": "pdf"},
    )

    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=0).chunk(document)

    assert chunks
    last_metadata = chunks[-1].metadata
    assert last_metadata["chapter_number"] == "4"
    assert last_metadata["chapter_title"] == "Public Finance"
    assert last_metadata["h1"] == "Chapter 4: Public Finance"
    assert last_metadata["h2"] == "4.1 Budget Systems"
    assert last_metadata["h3"] == "4.1.1 Procurement Controls"
    assert last_metadata["section_title"] == "4.1.1 Procurement Controls"


def test_payload_normalization_promotes_chapter_fields() -> None:
    payload = normalize_payload(
        "Chapter chunk text",
        {
            "source": "Data/report.pdf",
            "chunk_id": "chunk-1",
            "chapter_number": "7",
            "chapter_title": "Digital Infrastructure",
            "section_title": "7.2 Connectivity",
            "h1": "Chapter 7: Digital Infrastructure",
            "h2": "7.2 Connectivity",
        },
    ).to_qdrant_payload()

    assert payload["chapter_number"] == "7"
    assert payload["chapter_title"] == "Digital Infrastructure"
    assert payload["section_title"] == "7.2 Connectivity"
    assert payload["metadata"]["chapter_number"] == "7"


def test_qdrant_indexes_include_chapter_metadata() -> None:
    indexes = qdrant_payload_indexes()

    assert indexes["chapter_number"] == "keyword"
    assert indexes["chapter_title"] == "text"
    assert indexes["section_title"] == "text"
