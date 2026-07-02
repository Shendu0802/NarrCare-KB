"""FTS5 text index management."""
from src.db.connection import Database


class TextIndex:
    """Manages SQLite FTS5 full-text index."""

    def __init__(self, db: Database):
        self.db = db

    def rebuild(self) -> None:
        self.db.conn.execute("DELETE FROM knowledge_units_fts")
        rows = self.db.conn.execute(
            "SELECT rowid, title, text, summary FROM knowledge_units WHERE source_status IN ('main', 'candidate')"
        ).fetchall()
        for row in rows:
            self.db.conn.execute(
                "INSERT INTO knowledge_units_fts(rowid, title, text, summary) VALUES (?, ?, ?, ?)",
                (row["rowid"], row["title"], row["text"], row["summary"]),
            )
        self.db.conn.commit()

    def search(self, query: str, limit: int = 30) -> list[dict]:
        rows = self.db.conn.execute(
            """SELECT ku.id, ku.source_type, ku.source_status, ku.title,
                      snippet(knowledge_units_fts, 1, '<mark>', '</mark>', '...', 40) as snippet,
                      rank
               FROM knowledge_units_fts fts
               JOIN knowledge_units ku ON ku.rowid = fts.rowid
               WHERE knowledge_units_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
