"""FTS5 sparse recall with jieba tokenization."""
import jieba
from src.config import settings


class SparseRecaller:
    def __init__(self, text_index):
        self.text_index = text_index

    def recall(self, queries: list[str], top_k: int | None = None) -> list[dict]:
        if top_k is None:
            top_k = settings.sparse_top_k
        seen = set()
        results = []
        for query in queries:
            tokens = " ".join(jieba.cut(query))
            try:
                hits = self.text_index.search(tokens, limit=top_k)
            except Exception:
                hits = []
            for hit in hits:
                if hit["id"] not in seen:
                    seen.add(hit["id"])
                    hit["source"] = "sparse"
                    hit["score"] = float(hit.get("rank", 0.0))
                    results.append(hit)
        return results
