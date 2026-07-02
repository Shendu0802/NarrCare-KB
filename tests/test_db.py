import pytest
import os
import tempfile
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository


@pytest.fixture
def db():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.sqlite")
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()


class TestDatabase:
    def test_initialize_creates_tables(self, db):
        tables = db.list_tables()
        assert "knowledge_units" in tables
        assert "source_documents" in tables
        assert "eval_runs" in tables

    def test_fts_table_created(self, db):
        tables = db.list_tables()
        assert "knowledge_units_fts" in tables


class TestKnowledgeUnitRepository:
    def _make_data(self, id, status="main", text="content"):
        return {
            "id": id, "unit_type": "semantic_chunk", "source_type": "pdf_book",
            "source_status": status, "text": text, "semantic_tags": '[]',
            "scenario_tags": '[]', "role_tags": '[]', "method_tags": '[]',
            "risk_levels": '[]', "card_targets": '[]', "contraindications": '[]',
            "quality_flags": '[]', "parent_chunk_ids": '[]', "parse_method": "",
            "embedding_model": "", "created_at": "2026-07-01T00:00:00",
            "updated_at": "2026-07-01T00:00:00",
        }

    def test_insert_and_get(self, db):
        repo = KnowledgeUnitRepository(db)
        data = {**self._make_data("ku_test_001"), "title": "Test Chunk", "summary": "test summary", "quality_score": 0.9, "review_status": "unreviewed"}
        repo.insert(data)
        result = repo.get_by_id("ku_test_001")
        assert result is not None
        assert result["title"] == "Test Chunk"
        assert result["source_status"] == "main"

    def test_list_by_status(self, db):
        repo = KnowledgeUnitRepository(db)
        for i in range(3):
            repo.insert(self._make_data(f"ku_main_{i}", "main", f"content {i}"))
        for i in range(2):
            repo.insert(self._make_data(f"ku_cand_{i}", "candidate", f"c content {i}"))
        assert len(repo.list_by_status("main")) == 3
        assert len(repo.list_by_status("candidate")) == 2

    def test_get_all_active(self, db):
        repo = KnowledgeUnitRepository(db)
        repo.insert(self._make_data("ku_main", "main"))
        repo.insert(self._make_data("ku_cand", "candidate"))
        repo.insert(self._make_data("ku_quar", "quarantined"))
        active = repo.get_all_active()
        ids = [r["id"] for r in active]
        assert "ku_main" in ids
        assert "ku_cand" in ids
        assert "ku_quar" not in ids
