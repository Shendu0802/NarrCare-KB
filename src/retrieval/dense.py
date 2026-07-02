"""FAISS dense vector recall."""
from src.config import settings


class DenseRecaller:
    def __init__(self, embedder, vector_index):
        self.embedder = embedder
        self.index = vector_index

    def recall(self, queries: list[str], top_k: int | None = None) -> list[dict]:
        if top_k is None:
            top_k = settings.dense_top_k
        seen = set()
        results = []
        for query in queries:
            q_vec = self.embedder.encode_query(query)
            hits = self.index.search(q_vec, k=top_k)
            for hit in hits:
                if hit["id"] not in seen:
                    seen.add(hit["id"])
                    hit["source"] = "dense"
                    results.append(hit)
        return results
