from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from app.services.embeddings import HashEmbeddingModel


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9%]+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: list[dict[str, Any]] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avg_doc_len = 0.0

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._documents = documents
        self._term_freqs = []
        self._doc_freqs = Counter()
        total_len = 0
        for doc in documents:
            tokens = tokenize(doc["text"])
            total_len += len(tokens)
            freqs = Counter(tokens)
            self._term_freqs.append(freqs)
            for token in freqs:
                self._doc_freqs[token] += 1
        self._avg_doc_len = total_len / len(documents) if documents else 0.0

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._documents:
            return []
        query_tokens = tokenize(query)
        scores: list[tuple[int, float]] = []
        total_docs = len(self._documents)
        for idx, freqs in enumerate(self._term_freqs):
            doc_len = sum(freqs.values()) or 1
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self._doc_freqs[token]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                numerator = freqs[token] * (self.k1 + 1)
                denominator = freqs[token] + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1))
                score += idf * numerator / denominator
            scores.append((idx, score))
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                **self._documents[idx],
                "bm25_score": round(score, 4),
                "score": round(score, 4),
                "retrieval_source": "bm25",
            }
            for idx, score in ranked
            if score > 0
        ]


def reciprocal_rank_fusion(result_sets: list[list[dict[str, Any]]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            metadata = item.get("metadata") or {}
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue
            fused[chunk_id] += 1 / (k + rank)
    return fused


def rerank_results(query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(query))
    reranked = []
    for item in results:
        text_tokens = set(tokenize(item["text"]))
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        phrase_bonus = 0.15 if query.lower()[:30] in item["text"].lower() else 0
        domain_bonus = 0.1 if any(token in text_tokens for token in ["fee", "eligibility", "loan", "account", "transaction"]) else 0
        rerank_score = float(item.get("score", 0)) + overlap + phrase_bonus + domain_bonus
        reranked.append(
            {
                **item,
                "rerank_score": round(rerank_score, 4),
                "provenance": {
                    "retrieval_source": item.get("retrieval_source", "unknown"),
                    "vector_score": item.get("vector_score", 0),
                    "bm25_score": item.get("bm25_score", 0),
                    "rrf_score": item.get("rrf_score", 0),
                    "overlap_score": round(overlap, 4),
                    "domain_bonus": domain_bonus,
                },
            }
        )
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


def mmr_diversify(
    query: str,
    results: list[dict[str, Any]],
    top_k: int,
    lambda_mult: float = 0.72,
) -> list[dict[str, Any]]:
    if len(results) <= top_k:
        return results
    embedder = HashEmbeddingModel()
    query_vec = np.array(embedder.embed([query])[0])
    doc_vecs = [np.array(vec) for vec in embedder.embed([item["text"] for item in results])]
    selected: list[int] = []
    candidates = set(range(len(results)))
    while candidates and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")
        for idx in candidates:
            relevance = float(doc_vecs[idx] @ query_vec)
            diversity_penalty = max((float(doc_vecs[idx] @ doc_vecs[chosen]) for chosen in selected), default=0.0)
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    diversified = []
    for rank, idx in enumerate(selected, start=1):
        item = results[idx]
        diversified.append(
            {
                **item,
                "mmr_rank": rank,
                "provenance": {
                    **item.get("provenance", {}),
                    "mmr_lambda": lambda_mult,
                    "mmr_rank": rank,
                },
            }
        )
    return diversified


def compress_context(query: str, text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    query_tokens = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = set(tokenize(sentence))
        score = len(tokens & query_tokens) / max(len(query_tokens), 1)
        if any(term in sentence.lower() for term in ["must", "fee", "eligible", "consent", "audit", "approval"]):
            score += 0.2
        scored.append((score, sentence))
    selected: list[str] = []
    for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True):
        if sum(len(part) for part in selected) + len(sentence) + 1 > max_chars:
            continue
        selected.append(sentence)
        if sum(len(part) for part in selected) >= max_chars * 0.9:
            break
    return " ".join(selected)[:max_chars]
