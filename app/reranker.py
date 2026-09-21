from __future__ import annotations

import logging
import os
import threading
from typing import List, Sequence

import torch
from langchain_core.documents import Document

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    import logging
    logging.getLogger("pydantic_ai").warning("Failed to import transformers (perhaps missing libgthread-2.0.so.0): %s", e)
    AutoModelForSequenceClassification, AutoTokenizer = None, None

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)

from app.embeddings import BGE_CACHE_FOLDER


RERANKER_MODEL = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
RERANKER_DEVICE = (
    "cuda"
    if torch.cuda.is_available() and os.getenv("RERANKER_DEVICE", "auto").lower() != "cpu"
    else "cpu"
)

_RERANKER_SINGLETON: TransformersReranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker_singleton(model_name: str = RERANKER_MODEL) -> TransformersReranker:
    """Thread-safe module-level singleton accessor for TransformersReranker."""
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is None or getattr(_RERANKER_SINGLETON, "model_name", None) != model_name:
        with _RERANKER_LOCK:
            if _RERANKER_SINGLETON is None or getattr(_RERANKER_SINGLETON, "model_name", None) != model_name:
                _RERANKER_SINGLETON = TransformersReranker(model_name=model_name)
    return _RERANKER_SINGLETON



class TransformersReranker:
    """Pure Transformers reranker.

    This intentionally avoids external reranker wrappers to bypass tokenizer
    compatibility issues in the query/RAG runtime.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers module is not available due to a missing dependency (e.g. libgthread-2.0.so.0).")
        try:
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
            max_pos = getattr(self._model.config, "max_position_embeddings", 512)
            self.max_length = min(RERANKER_MAX_LENGTH, max_pos)
        except Exception as e:
            logging.warning("Failed to load reranker model '%s' (%s). Trying lightweight fallback 'cross-encoder/ms-marco-MiniLM-L-6-v2'...", model_name, e)
            try:
                fallback_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.model_name = fallback_name
                self._tokenizer = AutoTokenizer.from_pretrained(fallback_name, cache_dir=BGE_CACHE_FOLDER)
                self._model = AutoModelForSequenceClassification.from_pretrained(fallback_name, cache_dir=BGE_CACHE_FOLDER).to(RERANKER_DEVICE)
                self._model.eval()
                self.max_length = 512
                logging.info("Successfully loaded lightweight reranker fallback 'cross-encoder/ms-marco-MiniLM-L-6-v2'.")
            except Exception as e2:
                logging.error("Failed to load fallback reranker (%s). Using dummy pass-through scorer to prevent server crash.", e2)
                self._tokenizer = None
                self._model = None
                self.max_length = 512

    def _score_pairs_api(self, query: str, documents: list[str]) -> list[float] | None:
        """Score via free Hugging Face Serverless Inference API (0 MB local RAM)."""
        import requests
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": documents
            }
        }
        try:
            res = requests.post(api_url, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list):
                    # Format: [{"score": 0.9, ...}, ...] or [0.9, 0.5, ...]
                    if data and isinstance(data[0], dict) and "score" in data[0]:
                        # Sorting might be returned by rank index, align with input order if indexed
                        scores = [0.0] * len(documents)
                        for item in data:
                            idx = item.get("corpus_id", item.get("index"))
                            if idx is not None and idx < len(scores):
                                scores[idx] = float(item["score"])
                            elif "score" in item:
                                return [float(x["score"]) for x in data]
                        return scores
                    elif data and isinstance(data[0], (int, float)):
                        return [float(x) for x in data]
        except Exception as e:
            logging.debug("HF Serverless API rerank attempt failed: %s", e)
        return None

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        queries = [query for query, _document in pairs]
        documents = [document for _query, document in pairs]

        # 1. Try free Hugging Face Serverless API first (0 MB RAM)
        if queries:
            api_scores = self._score_pairs_api(queries[0], documents)
            if api_scores is not None and len(api_scores) == len(documents):
                return api_scores

        # 2. Local PyTorch model fallback
        if self._tokenizer is None or self._model is None:
            return [1.0] * len(pairs)

        encoded = self._tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
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
