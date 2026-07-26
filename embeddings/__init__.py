"""BGE-M3 embedding infrastructure for enriched conversational RAG chunks."""

from embeddings.embed_chunks import ChunkEmbedder, EmbeddedChunk
from embeddings.embedding_model import BgeM3EmbeddingModel, EmbeddingModelSettings, get_embedding_model
from embeddings.vector_store import FaissVectorStore, RetrievalResult

__all__ = [
    "BgeM3EmbeddingModel",
    "ChunkEmbedder",
    "EmbeddedChunk",
    "EmbeddingModelSettings",
    "FaissVectorStore",
    "RetrievalResult",
    "get_embedding_model",
]
