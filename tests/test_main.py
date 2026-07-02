from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestApp:
    def test_app_title(self):
        assert app.title == "NarrCare-KB"

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/health" in paths
