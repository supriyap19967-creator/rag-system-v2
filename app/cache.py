import math
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional


@dataclass
class CacheEntry:
    query: str
    embedding: List[float]
    response: Dict[str, object]


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self._similarity_threshold = similarity_threshold
        self._entries: List[CacheEntry] = []
        self._lock = Lock()

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def get(self, embedding: List[float]) -> Optional[Dict[str, object]]:
        with self._lock:
            best_score = 0.0
            best_response: Optional[Dict[str, object]] = None
            for entry in self._entries:
                score = self._cosine_similarity(entry.embedding, embedding)
                if score >= self._similarity_threshold and score > best_score:
                    best_score = score
                    best_response = dict(entry.response)
            return best_response

    def set(self, query: str, embedding: List[float], response: Dict[str, object]) -> None:
        with self._lock:
            self._entries.append(
                CacheEntry(
                    query=query,
                    embedding=list(embedding),
                    response=dict(response),
                )
            )
