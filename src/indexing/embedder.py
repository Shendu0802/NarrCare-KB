"""Embedding model wrapper for Qwen3-Embedding-0.6B."""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps sentence-transformers for batch encoding and single-query encoding."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> bool:
        try:
            self._model = SentenceTransformer(self.model_path, device=self.device)
            if self.device == "cuda":
                self._model.half()
            return True
        except Exception:
            self._model = None
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Embedder model not loaded")
        return self._model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], batch_size=1)[0]
