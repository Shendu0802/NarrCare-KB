#!/usr/bin/env python3
"""Batch import local files into NarrCare-KB."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db.connection import Database
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker
from src.ingestion.orchestrator import IngestionOrchestrator


def main():
    import_dir = "参考资料"
    if not os.path.isdir(import_dir):
        print(f"Directory not found: {import_dir}")
        sys.exit(1)

    pdf_files = [
        os.path.join(import_dir, f)
        for f in os.listdir(import_dir)
        if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF files to import.")

    db = Database(settings.db_path)
    db.initialize()
    orch = IngestionOrchestrator(
        db=db,
        parser=DocumentParser(),
        cleaner=TextCleaner(),
        chunker=SemanticChunker(
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
    )

    for fp in pdf_files:
        print(f"\nImporting: {fp}")
        result = orch.ingest_file(fp, source_status="main")
        print(f"  Status: {result['status']}")
        print(f"  Chunks: {result.get('chunk_count', 0)}")
        print(f"  Quarantined: {result.get('quarantined_count', 0)}")

    db.close()
    print("\nDone. Run POST /index/rebuild to build search indexes.")


if __name__ == "__main__":
    main()
