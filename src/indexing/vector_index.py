"""FAISS-based dense vector index with persistence."""
import os
import json
import numpy as np
import faiss


class VectorIndex:
    """FAISS-based dense vector index with persistence."""

    def __init__(self, dim: int = 1024, index_path: str = "data/index/faiss.index"):
        self.dim = dim
        self.index_path = index_path
        self.index: faiss.Index | None = None
        self.id_map: dict[int, str] = {}

    def build(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.id_map = {i: cid for i, cid in enumerate(chunk_ids)}

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[dict]:
        if self.index is None:
            return []
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                results.append({"id": self.id_map[idx], "score": float(score)})
        return results

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        map_path = self.index_path.replace(".index", "_id_map.json")
        with open(map_path, "w") as f:
            json.dump(self.id_map, f)
        meta_path = self.index_path.replace(".index", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"dim": self.dim, "size": self.index.ntotal}, f)

    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            return False
        try:
            self.index = faiss.read_index(self.index_path)
            map_path = self.index_path.replace(".index", "_id_map.json")
            with open(map_path) as f:
                self.id_map = {int(k): v for k, v in json.load(f).items()}
            return True
        except Exception:
            return False

    def add(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        if self.index is None:
            self.build(chunk_ids, embeddings)
            return
        start_idx = self.index.ntotal
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for i, cid in enumerate(chunk_ids):
            self.id_map[start_idx + i] = cid

    @property
    def is_loaded(self) -> bool:
        return self.index is not None

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
