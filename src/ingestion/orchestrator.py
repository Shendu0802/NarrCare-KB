"""Orchestrates the full ingestion pipeline."""
import hashlib
import uuid
import os
from datetime import datetime, timezone
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker


class IngestionOrchestrator:
    """Coordinates the full ingestion pipeline: parse → clean → chunk → store."""

    def __init__(self, db: Database, parser: DocumentParser, cleaner: TextCleaner, chunker: SemanticChunker):
        self.db = db
        self.parser = parser
        self.cleaner = cleaner
        self.chunker = chunker
        self.repo = KnowledgeUnitRepository(db)

    def _file_hash(self, file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _is_already_imported(self, file_path: str) -> bool:
        row = self.db.conn.execute(
            "SELECT id FROM source_documents WHERE file_path = ? LIMIT 1", (file_path,)
        ).fetchone()
        return row is not None

    def ingest_file(self, file_path: str, source_type: str | None = None, source_status: str = "main") -> dict:
        """Run the full ingestion pipeline on a single file."""
        if not os.path.exists(file_path):
            return {"status": "failed", "message": f"File not found: {file_path}", "chunk_count": 0, "quarantined_count": 0}

        if self._is_already_imported(file_path):
            return {"status": "skipped", "message": f"File already imported: {file_path}", "chunk_count": 0, "quarantined_count": 0}

        if source_type is None:
            source_type = self.parser.detect_source_type(file_path)

        now = datetime.now(timezone.utc).isoformat()
        source_title = os.path.basename(file_path)
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        self.db.conn.execute(
            """INSERT OR REPLACE INTO source_documents (id, file_path, source_type, title, parse_status, imported_at)
               VALUES (?, ?, ?, ?, 'parsing', ?)""",
            (doc_id, file_path, source_type, source_title, now),
        )
        self.db.conn.commit()

        try:
            parsed_doc = self.parser.parse(file_path, source_type)
            cleaned_pages = [self.cleaner.clean(page) for page in parsed_doc.pages]
            chunks = self.chunker.chunk(cleaned_pages, source_title=source_title, source_uri=file_path, source_type=source_type)

            chunk_count = 0
            quarantined_count = 0
            for chunk in chunks:
                quality_score = 0.8
                quality_flags = "[]"
                parse_method = chunk.get("parse_method", "text_layer")
                final_status = self.cleaner.determine_status(quality_score, source_status)
                if final_status == "quarantined":
                    quarantined_count += 1
                else:
                    chunk_count += 1

                self.repo.insert({
                    "id": chunk["id"], "unit_type": chunk["unit_type"],
                    "source_type": chunk["source_type"], "source_status": final_status,
                    "title": chunk.get("title", ""), "text": chunk["text"], "summary": "",
                    "source_uri": chunk.get("source_uri", ""),
                    "source_title": chunk.get("source_title", ""), "source_citation": "",
                    "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"),
                    "semantic_tags": "[]", "scenario_tags": "[]", "role_tags": "[]",
                    "method_tags": "[]", "risk_levels": "[]", "card_targets": "[]",
                    "contraindications": "[]", "quality_score": quality_score,
                    "quality_flags": quality_flags, "review_status": "unreviewed",
                    "parent_chunk_ids": "[]", "parse_method": parse_method,
                    "embedding_model": "", "created_at": now, "updated_at": now,
                })

            self.db.conn.execute(
                "UPDATE source_documents SET parse_status = 'done', chunk_count = ? WHERE id = ?",
                (chunk_count, doc_id),
            )
            self.db.conn.commit()

            return {
                "status": "done", "message": f"Ingested {len(chunks)} chunks from {source_title}",
                "document_id": doc_id, "chunk_count": chunk_count,
                "total_segments": len(chunks), "quarantined_count": quarantined_count,
            }
        except Exception as e:
            self.db.conn.execute(
                "UPDATE source_documents SET parse_status = 'failed' WHERE id = ?", (doc_id,)
            )
            self.db.conn.commit()
            return {"status": "failed", "message": str(e), "chunk_count": 0, "quarantined_count": 0}

    def ingest_batch(self, file_paths: list[str], source_type: str | None = None, source_status: str = "main") -> list[dict]:
        return [self.ingest_file(fp, source_type=source_type, source_status=source_status) for fp in file_paths]
