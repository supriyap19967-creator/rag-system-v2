from __future__ import annotations

import logging
import os
from typing import List, Sequence

import torch
from langchain_core.documents import Document

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)

from app.embeddings import BGE_CACHE_FOLDER


RERANKER_MODEL = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
RERANKER_DEVICE = (
    "cuda"
    if torch.cuda.is_available() and os.getenv("RERANKER_DEVICE", "auto").lower() != "cpu"
    else "cpu"
)


class TransformersReranker:
    """Pure Transformers reranker.

    This intentionally avoids external reranker wrappers to bypass tokenizer
    compatibility issues in the query/RAG runtime.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=BGE_CACHE_FOLDER,
            trust_remote_code=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=BGE_CACHE_FOLDER,
            trust_remote_code=True,
        ).to(RERANKER_DEVICE)
        self._model.eval()

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        queries = [query for query, _document in pairs]
        documents = [document for _query, document in pairs]
        encoded = self._tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(RERANKER_DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                scores = logits[:, 0]
            elif logits.ndim == 2:
                scores = logits[:, -1]
            else:
                scores = logits.reshape(-1)
        return [float(score) for score in scores.detach().cpu().tolist()]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score query/document pairs without relying on reranker wrapper packages."""

        return self._score_pairs(pairs)

    def rerank(self, query: str, documents: Sequence[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, document.page_content) for document in documents]
        scores = self._score_pairs(pairs)

        reranked_documents = []
        for document, score in zip(documents, scores):
            enriched_metadata = dict(document.metadata)
            enriched_metadata["rerank_score"] = float(score)
            reranked_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata=enriched_metadata,
                )
            )

        reranked_documents.sort(
            key=lambda document: float(document.metadata.get("rerank_score", 0.0)),
            reverse=True,
        )
        return reranked_documents[:top_k]
