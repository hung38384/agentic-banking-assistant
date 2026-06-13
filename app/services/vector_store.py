from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.services.documents import DocumentChunk, chunk_documents, load_markdown_documents
from app.services.embeddings import HashEmbeddingModel
from app.services.retrieval import BM25Index, mmr_diversify, reciprocal_rank_fusion, rerank_results


class BankingVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_model = HashEmbeddingModel()
        self._collection: Any | None = None
        self._local_chunks: list[DocumentChunk] = []
        self._local_embeddings: list[list[float]] = []
        self._bm25 = BM25Index()
        self.backend = "local"
        self._init_backend()

    def _init_backend(self) -> None:
        if self.settings.vector_backend.lower() != "chroma":
            return
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.settings.chroma_dir)
            self._collection = client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chroma"
        except Exception:
            self._collection = None
            self.backend = "local"

    def ingest_path(self, docs_dir: Path, reset: bool = True) -> tuple[int, int]:
        docs = load_markdown_documents(docs_dir)
        chunks = chunk_documents(docs)
        self.upsert(chunks, reset=reset)
        return len(docs), len(chunks)

    def upsert(self, chunks: list[DocumentChunk], reset: bool = True) -> None:
        if reset:
            self._local_chunks = []
            self._local_embeddings = []
            self._bm25.build([])
            if self._collection is not None:
                try:
                    existing = self._collection.get(include=[])
                    ids = existing.get("ids", [])
                    if ids:
                        self._collection.delete(ids=ids)
                except Exception:
                    pass

        embeddings = self.embedding_model.embed([chunk.text for chunk in chunks])
        self._local_chunks.extend(chunks)
        self._local_embeddings.extend(embeddings)
        self._rebuild_bm25()

        if self._collection is not None:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=embeddings,
            )

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.hybrid_search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.settings.top_k
        candidate_k = max(k, self.settings.rerank_top_n)
        vector_results = self._vector_search(query, candidate_k)
        bm25_results = self._bm25.search(query, candidate_k)
        fused_scores = reciprocal_rank_fusion([vector_results, bm25_results])
        by_chunk_id: dict[str, dict[str, Any]] = {}
        for item in [*vector_results, *bm25_results]:
            metadata = item.get("metadata") or {}
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue
            current = by_chunk_id.get(chunk_id, {})
            by_chunk_id[chunk_id] = {
                **item,
                "vector_score": max(float(current.get("vector_score", 0)), float(item.get("vector_score", item.get("score", 0))) if item.get("retrieval_source") != "bm25" else float(current.get("vector_score", 0))),
                "bm25_score": max(float(current.get("bm25_score", 0)), float(item.get("bm25_score", 0))),
                "rrf_score": fused_scores.get(chunk_id, 0.0),
            }
        weighted = []
        for item in by_chunk_id.values():
            score = (
                self.settings.hybrid_vector_weight * float(item.get("vector_score", 0))
                + self.settings.hybrid_bm25_weight * float(item.get("bm25_score", 0))
                + float(item.get("rrf_score", 0))
            )
            weighted.append({**item, "score": round(score, 4), "retrieval_source": "hybrid"})
        reranked = rerank_results(query, weighted, top_k=max(k, min(len(weighted), self.settings.rerank_top_n)))
        return mmr_diversify(query, reranked, top_k=k)

    def _vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if self._collection is not None:
            result = self._collection.query(
                query_embeddings=self.embedding_model.embed([query]),
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            docs = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                {
                    "text": doc,
                    "metadata": metadata,
                    "score": round(1 - float(distance), 4),
                    "vector_score": round(1 - float(distance), 4),
                    "retrieval_source": "vector",
                }
                for doc, metadata, distance in zip(docs, metadatas, distances, strict=False)
            ]
        return self._local_search(query, k)

    def _local_search(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self._local_chunks:
            return []
        query_vec = np.array(self.embedding_model.embed([query])[0])
        matrix = np.array(self._local_embeddings)
        scores = matrix @ query_vec
        ranked = np.argsort(scores)[::-1][:k]
        return [
            {
                "text": self._local_chunks[idx].text,
                "metadata": asdict(self._local_chunks[idx])["metadata"],
                "score": round(float(scores[idx]), 4),
                "vector_score": round(float(scores[idx]), 4),
                "retrieval_source": "vector",
            }
            for idx in ranked
        ]

    def _rebuild_bm25(self) -> None:
        documents = [
            {
                "text": chunk.text,
                "metadata": asdict(chunk)["metadata"],
                "score": 0.0,
                "retrieval_source": "bm25",
            }
            for chunk in self._local_chunks
        ]
        self._bm25.build(documents)
