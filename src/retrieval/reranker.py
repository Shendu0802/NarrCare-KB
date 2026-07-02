"""Qwen3 Reranker for candidate re-ranking."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import settings


class Qwen3Reranker:
    """Qwen3-Reranker-0.6B cross-encoder for relevance scoring."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> bool:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
            return True
        except Exception:
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, candidates: list[dict], top_k: int = 20) -> list[dict]:
        if not self.is_loaded or not candidates:
            return candidates

        pairs = []
        for c in candidates:
            unit_text = c.get("text", c.get("snippet", ""))[:512]
            pairs.append(f"Query: {query}\nDocument: {unit_text}")

        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self._model(**inputs).logits.squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["score"] = (
                float(score) * settings.weight_rerank
                + candidates[i].get("score", 0.0)
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
