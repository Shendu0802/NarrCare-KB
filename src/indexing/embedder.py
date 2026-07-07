"""Embedding module — supports local TF-IDF and API-based embeddings.

Default: sklearn TfidfVectorizer (zero extra deps, fast, works everywhere).
Future: swap to Qwen3-Embedding or API embeddings by changing the factory.
"""
import numpy as np
import pickle
import os


class Embedder:
    """Text vectorizer.

    Uses sklearn TfidfVectorizer by default for immediate availability.
    Set model_path to 'api' to use OpenAI-compatible API embeddings instead.
    """

    def __init__(self, model_path: str = "tfidf", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._vectorizer = None
        self._dim = None
        self._fitted = False

    def load(self) -> bool:
        if self.model_path == "api":
            return self._load_api()
        return self._load_tfidf()

    def _load_tfidf(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=1024,
                analyzer='char_wb',
                ngram_range=(2, 4),
            )
            return True
        except Exception:
            return False

    def _load_api(self) -> bool:
        try:
            from openai import OpenAI
            from src.config import settings
            self._client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key or None,
                timeout=settings.llm_timeout,
            )
            self._api_model = "text-embedding-v3"  # Qwen embedding model
            self._api_dim = 1024
            return True
        except Exception:
            return False

    @property
    def is_loaded(self) -> bool:
        return self._vectorizer is not None or hasattr(self, '_client')

    def encode(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        if hasattr(self, '_client'):
            return self._encode_api(texts, batch_size)

        if not self._fitted:
            vectors = self._vectorizer.fit_transform(texts).toarray().astype(np.float32)
            self._fitted = True
            self._dim = vectors.shape[1]
        else:
            vectors = self._vectorizer.transform(texts).toarray().astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _encode_api(self, texts: list[str], batch_size: int) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(
                model=self._api_model, input=batch,
                dimensions=getattr(self, '_api_dim', 1024),
            )
            for item in resp.data:
                all_embs.append(np.array(item.embedding, dtype=np.float32))
        result = np.stack(all_embs)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / norms

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], batch_size=1)[0]

    def save(self, path: str) -> None:
        """Save fitted vectorizer for later query encoding."""
        if self._vectorizer is not None and self._fitted:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self._vectorizer, f)

    def load_vectorizer(self, path: str) -> bool:
        """Load a previously saved fitted vectorizer."""
        try:
            with open(path, 'rb') as f:
                self._vectorizer = pickle.load(f)
            self._fitted = True
            return True
        except Exception:
            return False
