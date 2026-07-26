from gateway_guardrails import GatewayInfrastructure
from vectordb.retrieval_pipeline import ConversationalRetrievalPipeline


def test_conversational_retrieval_masks_entire_history_payload() -> None:
    pipeline = ConversationalRetrievalPipeline.__new__(ConversationalRetrievalPipeline)
    pipeline.max_history_turns = 4
    pipeline.gateway = GatewayInfrastructure(request_cap=1_000_000)

    payload = pipeline._query_with_context(
        "Compare it with analyst@example.com",
        [
            {"role": "user", "content": "My phone is +1 (415) 555-0134. Summarize Figure 8.4."},
            {"role": "assistant", "content": "Figure 8.4 discusses the requested topic."},
        ],
    )

    assert "+1 (415) 555-0134" not in payload
    assert "analyst@example.com" not in payload
    assert "[REDACTED_PHONE]" in payload
    assert "[REDACTED_EMAIL]" in payload


def test_gateway_mask_pii_handles_aggregated_multiturn_text() -> None:
    gateway = GatewayInfrastructure(request_cap=1_000_000)
    aggregated_payload = (
        "Conversation history:\n"
        "user: My phone is +1 (415) 555-0134.\n\n"
        "Latest user message:\n"
        "Email me at analyst@example.com"
    )

    masked = gateway.mask_pii(aggregated_payload)

    assert "+1 (415) 555-0134" not in masked
    assert "analyst@example.com" not in masked
    assert "[REDACTED_PHONE]" in masked
    assert "[REDACTED_EMAIL]" in masked
