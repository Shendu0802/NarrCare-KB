import os
import pytest
from src.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.embedding_model == "Qwen3-Embedding-0.6B"
        assert settings.reranker_model == "Qwen3-Reranker-0.6B"
        assert settings.chunk_min_chars == 300
        assert settings.chunk_max_chars == 800
        assert settings.weight_rerank == 0.55

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("KB_PORT", "9999")
        monkeypatch.setenv("KB_LLM_MODEL", "gpt-4")
        settings = Settings()
        assert settings.port == 9999
        assert settings.llm_model == "gpt-4"

    def test_weight_sum(self):
        settings = Settings()
        assert all(w > 0 for w in [
            settings.weight_rerank, settings.weight_recall,
            settings.weight_metadata, settings.weight_quality,
            settings.weight_source_status
        ])
