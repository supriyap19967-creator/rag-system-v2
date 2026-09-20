from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Dict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from app.reranker import TransformersReranker
from langchain_core.documents import Document

# Load env variables
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "rag_master_v2"

class ResearchOutput(BaseModel):
    executive_summary: str = Field(
        description="A concise 2–3 sentence overview answering the core request."
    )
    key_findings: List[str] = Field(
        description="Detailed breakdown using bullet points, tables, or categorized sections."
    )
    source_notes: str = Field(
        description="Any caveats, data limitations, source details, page numbers, or file references."
    )

# Setup Research Agent
model_name = os.getenv("RESEARCH_MODEL_NAME", "groq:llama-3.3-70b-versatile")
research_agent = Agent(
    model_name,
    output_type=ResearchOutput,
    retries=3,
    system_prompt=(
        "You are an expert analytical research assistant. Your task is to provide a complete, well-structured, and balanced answer based ONLY on the provided context chunks.\n\n"
        "INSTRUCTIONS:\n"
        "- Comprehensive Coverage: Address all facets, conditions, sub-points, and nuances explicitly mentioned in the context regarding the user's question.\n"
        "- Strict Factual Grounding: Include specific entities, metrics, definitions, and policy mechanisms directly from the source material. Do not summarize so aggressively that key technical details are lost.\n"
        "- No Asset Hallucination: Never invent, guess, or generalize asset or image filenames (like 'workers.png'). Only reference exact filenames (e.g. 'world_development_report_2025_figure_7_2.png') provided in the retrieved chunks context.\n"
        "- Structural Clarity: Organize the response logically using concise bullet points, bold key concepts, or light Markdown tables where appropriate.\n"
        "- No Filler: Avoid preamble, filler, generic conclusions, or speculation outside the retrieved context.\n"
        "- Enumerate Systematic Perspectives: If the context contains multiple perspectives, requirements, or steps, enumerate them systematically."
    )
)

@research_agent.tool
def search_knowledge_base(ctx: RunContext[Any], query: str) -> str:
    """
    Search the hybrid Qdrant vector store using Reciprocal Rank Fusion (RRF)
    and Cross-Encoder reranking to find relevant document text chunks.
    """
    logger.info(f"Research Agent searching KB for: '{query}'")
    
    # Initialize Qdrant Client local
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    if not client.collection_exists(COLLECTION_NAME):
        return f"Error: Qdrant collection '{COLLECTION_NAME}' does not exist."
        
    try:
        # Generate Hybrid Embeddings
        # Dynamically resolve vector space names
        try:
            coll_info = client.get_collection(collection_name=COLLECTION_NAME)
            vec_names = list(coll_info.config.params.vectors.keys()) if isinstance(coll_info.config.params.vectors, dict) else ["dense"]
        except Exception:
            vec_names = ["dense"]
            
        dense_using = "text-dense" if "text-dense" in vec_names else "dense"
        sparse_using = "text-sparse" if "text-sparse" in vec_names else "sparse"
        
        # Generate Hybrid Embeddings
        from embeddings.embedding_model import get_embedding_model
        dense_model = get_embedding_model()
        dense_vec = dense_model.embed_query(query)
        
        sparse_model = SparseTextEmbedding("Qdrant/bm25")
        
        qdrant_filter = None
        try:
            from app.main import extract_chapter_references, build_chapter_filter
            chapter_numbers = extract_chapter_references(query)
            if chapter_numbers:
                qdrant_filter = build_chapter_filter(chapter_numbers)
        except Exception:
            pass
        
        sparse_vec = list(sparse_model.embed([query]))[0]
        
        qdrant_sparse_vec = models.SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist()
        )
        
        # Parallel Prefetch Search using Qdrant Native RRF API
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=dense_using,
                    filter=qdrant_filter,
                    limit=20
                ),
                models.Prefetch(
                    query=qdrant_sparse_vec,
                    using=sparse_using,
                    filter=qdrant_filter,
                    limit=20
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=15
        )
        raw_results = response.points
        
        # Print/log search results size and scores
        logger.info("Retrieved %s raw search results from Qdrant in research agent", len(raw_results))
        print(f"DEBUG: Retrieved {len(raw_results)} raw search results from Qdrant in research agent.")
        for idx, pt in enumerate(raw_results, start=1):
            logger.info("  [%s] Point ID: %s, Score: %s", idx, pt.id, pt.score)
            print(f"  [{idx}] Point ID: {pt.id}, Score: {pt.score}")
        
        if not raw_results:
            return "No matching context fragments returned from Qdrant vector store."
            
        # Cross-Encoder Re-ranking Pipeline
        documents = []
        for point in raw_results:
            payload = point.payload or {}
            text = (payload.get("text") or payload.get("page_content") or "").strip()
            documents.append(Document(page_content=text, metadata=payload.get("metadata", {})))
            
        reranker = TransformersReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked_docs = reranker.rerank(query, documents, top_k=3)
        
        # Format returned chunks
        formatted_chunks = []
        for index, doc in enumerate(reranked_docs, start=1):
            metadata = doc.metadata
            source = metadata.get("source_file", "unknown")
            page = metadata.get("page_number", "N/A")
            formatted_chunks.append(
                f"[{index}] Source: {source} (Pg: {page})\nContent: {doc.page_content.strip()}"
            )
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        logger.exception("Search failed")
        return f"Error during search execution: {e}"

def execute_research(query: str) -> Dict[str, Any]:
    """
    Entrypoint function to execute research query.
    """
    run_result = research_agent.run_sync(query)
    output: ResearchOutput = run_result.output
    return {
        "executive_summary": output.executive_summary,
        "key_findings": output.key_findings,
        "source_notes": output.source_notes
    }
