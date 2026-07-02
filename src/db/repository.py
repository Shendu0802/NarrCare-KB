"""Repository pattern for KnowledgeUnit CRUD operations."""
from src.db.connection import Database


class KnowledgeUnitRepository:
    def __init__(self, db: Database):
        self.db = db

    def insert(self, data: dict) -> None:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT OR REPLACE INTO knowledge_units ({columns}) VALUES ({placeholders})"
        self.db.conn.execute(sql, list(data.values()))
        self.db.conn.commit()

    def get_by_id(self, ku_id: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT * FROM knowledge_units WHERE id = ?", (ku_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_by_status(self, source_status: str) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT * FROM knowledge_units WHERE source_status = ?", (source_status,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_active(self) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT * FROM knowledge_units WHERE source_status IN ('main', 'candidate')"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self.db.conn.execute(
            "SELECT source_status, COUNT(*) as cnt FROM knowledge_units GROUP BY source_status"
        ).fetchall()
        return {r["source_status"]: r["cnt"] for r in rows}

    def count_by_type(self) -> dict[str, int]:
        rows = self.db.conn.execute(
            "SELECT unit_type, COUNT(*) as cnt FROM knowledge_units GROUP BY unit_type"
        ).fetchall()
        return {r["unit_type"]: r["cnt"] for r in rows}

    def search_by_tags(self, tags: list[str], field: str = "scenario_tags", limit: int = 20) -> list[dict]:
        if not tags:
            return []
        conditions = " OR ".join(f"{field} LIKE ?" for _ in tags)
        params = [f"%{t}%" for t in tags]
        rows = self.db.conn.execute(
            f"SELECT * FROM knowledge_units WHERE ({conditions}) AND source_status IN ('main', 'candidate') LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def fts_search(self, query: str, limit: int = 30) -> list[dict]:
        rows = self.db.conn.execute(
            """SELECT ku.* FROM knowledge_units ku
               INNER JOIN knowledge_units_fts fts ON ku.rowid = fts.rowid
               WHERE knowledge_units_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
