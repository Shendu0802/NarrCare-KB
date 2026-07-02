"""SQLite connection management and schema initialization."""
import sqlite3
import os


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_units (
    id              TEXT PRIMARY KEY,
    unit_type       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_status   TEXT NOT NULL DEFAULT 'candidate',
    title           TEXT DEFAULT '',
    text            TEXT NOT NULL,
    summary         TEXT DEFAULT '',
    source_uri      TEXT DEFAULT '',
    source_title    TEXT DEFAULT '',
    source_citation TEXT DEFAULT '',
    page_start      INTEGER,
    page_end        INTEGER,
    semantic_tags   TEXT DEFAULT '[]',
    scenario_tags   TEXT DEFAULT '[]',
    role_tags       TEXT DEFAULT '[]',
    method_tags     TEXT DEFAULT '[]',
    risk_levels     TEXT DEFAULT '[]',
    card_targets    TEXT DEFAULT '[]',
    contraindications TEXT DEFAULT '[]',
    quality_score   REAL NOT NULL DEFAULT 0.0,
    quality_flags   TEXT DEFAULT '[]',
    review_status   TEXT NOT NULL DEFAULT 'unreviewed',
    parent_chunk_ids TEXT DEFAULT '[]',
    parse_method    TEXT DEFAULT '',
    embedding_model TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_units_fts USING fts5(
    title, text, summary,
    content='knowledge_units',
    content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS source_documents (
    id              TEXT PRIMARY KEY,
    file_path       TEXT,
    source_uri      TEXT,
    source_type     TEXT NOT NULL,
    title           TEXT DEFAULT '',
    total_pages     INTEGER,
    parse_status    TEXT NOT NULL DEFAULT 'pending',
    chunk_count     INTEGER DEFAULT 0,
    imported_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              TEXT PRIMARY KEY,
    run_at          TEXT NOT NULL,
    total_queries   INTEGER,
    recall_at_10    REAL,
    safety_hit_rate REAL,
    noise_rate      REAL,
    avg_usability   REAL,
    details_json    TEXT DEFAULT '{}'
);
"""


class Database:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def initialize(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def list_tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
