from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.embeddings import HashEmbeddingModel


@dataclass
class CacheEntry:
    key: str
    query: str
    response: dict[str, Any]
    embedding: list[float]


class SemanticCache:
    def __init__(self, threshold: float = 0.94):
        self.threshold = threshold
        self.embedding_model = HashEmbeddingModel()
        self._entries: list[CacheEntry] = []

    def get(self, query: str) -> dict[str, Any] | None:
        if not self._entries:
            return None
        query_vec = np.array(self.embedding_model.embed([query])[0])
        query_tokens = _tokens(query)
        best: tuple[float, CacheEntry] | None = None
        for entry in self._entries:
            lexical = _jaccard(query_tokens, _tokens(entry.query))
            if lexical < 0.65:
                continue
            score = float(np.array(entry.embedding) @ query_vec)
            if best is None or score > best[0]:
                best = (score, entry)
        if best and best[0] >= self.threshold:
            return {**best[1].response, "cache_hit": True, "cache_score": round(best[0], 4)}
        return None

    def set(self, query: str, response: dict[str, Any]) -> None:
        key = hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()[:16]
        embedding = self.embedding_model.embed([query])[0]
        self._entries.append(CacheEntry(key=key, query=query, response=response, embedding=embedding))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9%]+", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


semantic_cache = SemanticCache()
