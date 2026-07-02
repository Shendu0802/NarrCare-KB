import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "embedding_model" in data
        assert "reranker_model" in data
        assert "index_loaded" in data
        assert "index_version" in data
        assert "document_count" in data
        assert "chunk_count" in data
        assert "card_count" in data

    def test_health_status_is_valid(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("ok", "degraded", "error")
