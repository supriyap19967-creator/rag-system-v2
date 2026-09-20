from __future__ import annotations

import logging
import os
from typing import Literal

os.environ["QDRANT_PATH"] = os.path.abspath("./qdrant_db")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    asset_resolver_node,
    generator_node,
    retriever_node,
    router_node,
    verifier_node,
)
from app.graph.state import RAGState

logger = logging.getLogger(__name__)


def should_retry(state: RAGState) -> Literal["retriever", "__end__"]:
    """Self-healing conditional edge router."""
    is_grounded = state.get("is_grounded", False)
    retry_count = state.get("retry_count", 0)

    if not is_grounded and retry_count < 2:
        logger.info("LangGraph [Self-Healing Edge] Answer ungrounded. Routing back to Retriever (Retry %d)...", retry_count)
        return "retriever"

    logger.info("LangGraph [Workflow End Edge] Response verified successfully. Completing graph execution.")
    return END


def create_rag_graph():
    """Build and compile the LangGraph RAG StateGraph workflow."""
    workflow = StateGraph(RAGState)

    # 1. Add all 5 Graph Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("asset_resolver", asset_resolver_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("verifier", verifier_node)

    # 2. Add Sequential Edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "asset_resolver")
    workflow.add_edge("asset_resolver", "generator")
    workflow.add_edge("generator", "verifier")

    # 3. Add Self-Healing Conditional Retry Edge
    workflow.add_conditional_edges("verifier", should_retry, ["retriever", END])

    # 4. Attach Memory Checkpointer & Compile Graph
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph RAG StateGraph compiled successfully with MemorySaver checkpointer.")
    return compiled_graph


# Singleton compiled graph instance
rag_graph = create_rag_graph()
