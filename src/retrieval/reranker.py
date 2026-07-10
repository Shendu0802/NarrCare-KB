"""Reranker — API-based (DeepSeek) and local (Qwen3, needs GPU)."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import settings


class Qwen3Reranker:
    """Local Qwen3-Reranker-0.6B cross-encoder. Requires GPU with CC 7.5+."""
    # ... (unchanged, kept for future GPU availability)
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
        inputs = self._tokenizer(pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
        with torch.no_grad():
            scores = self._model(**inputs).logits.squeeze(-1).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["score"] = float(score) * settings.weight_rerank + candidates[i].get("score", 0.0)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


class APIReranker:
    """LLM API-based reranker — sends candidates to DeepSeek for relevance scoring.

    No GPU needed. Uses sync OpenAI client for simple thread-safe calling.
    """

    RERANK_PROMPT = """You are a clinical relevance judge. Rate each passage's relevance to the query on a scale of 0-10.
Query: {query}

Passages:
{passages}

Return JSON only with key "scores" mapping to a list of integers, one per passage in order."""

    def __init__(self, llm_client_or_config=None):
        from openai import OpenAI
        if llm_client_or_config is None:
            from src.config import settings
            self._client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout,
            )
            self._model = settings.llm_model
        elif hasattr(llm_client_or_config, 'config'):
            # LLMClient passed — extract config
            c = llm_client_or_config.config
            self._client = OpenAI(base_url=c.base_url, api_key=c.api_key, timeout=c.timeout)
            self._model = c.model
        else:
            self._client = OpenAI(
                base_url=llm_client_or_config.base_url,
                api_key=llm_client_or_config.api_key,
                timeout=llm_client_or_config.timeout,
            )
            self._model = llm_client_or_config.model

    def rerank(self, query: str, candidates: list[dict], top_k: int = 20) -> list[dict]:
        if not candidates:
            return candidates

        batch = candidates[:min(len(candidates), 20)]
        passages_text = ""
        for i, c in enumerate(batch):
            snippet = (c.get("text") or c.get("snippet", ""))[:200]
            passages_text += f"[{i}] {snippet}\n"

        prompt = self.RERANK_PROMPT.format(query=query[:300], passages=passages_text)

        try:
            import json as _json
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a clinical relevance judge. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or "{}"
            result = _json.loads(text)
            scores = result.get("scores", [])
        except Exception:
            return self._fallback_rerank(candidates, top_k)

        # Apply scores
        for i, score in enumerate(scores[:len(batch)]):
            batch[i]["rerank_score"] = float(score) / 10.0
            batch[i]["score"] = (
                float(score) / 10.0 * settings.weight_rerank
                + batch[i].get("score", 0.0)
            )

        batch.sort(key=lambda x: x["score"], reverse=True)
        return batch[:top_k]

    def _fallback_rerank(self, candidates: list[dict], top_k: int) -> list[dict]:
        """If API fails, return top-k by existing score."""
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates[:top_k]


# Factory
def create_reranker(llm_config=None):
    """Create the best available reranker. Uses API by default (no GPU needed)."""
    return APIReranker(llm_config)
