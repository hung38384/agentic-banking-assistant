import hashlib
import math
import re

import numpy as np


class HashEmbeddingModel:
    """Small deterministic embedding model for local demos and tests.

    Production deployments can replace this class with OpenAI, Gemini, or a
    sentence-transformers adapter while keeping the vector-store contract.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9%]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign * (1.0 + math.log1p(len(token)))
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector.tolist()

