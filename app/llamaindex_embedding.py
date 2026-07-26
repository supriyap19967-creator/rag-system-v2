from typing import List

from llama_index.core.embeddings import BaseEmbedding

from app.embeddings import BGE_MODEL_NAME, get_bge_embeddings


class BgeLlamaIndexEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter for the repo's existing BGE model."""

    def __init__(self, **kwargs):
        super().__init__(model_name=BGE_MODEL_NAME, **kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        return list(get_bge_embeddings().embed_query(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return list(get_bge_embeddings().embed_query(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [list(vector) for vector in get_bge_embeddings().embed_documents(texts)]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
