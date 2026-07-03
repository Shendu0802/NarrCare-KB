#!/usr/bin/env python3
"""Batch import local files into NarrCare-KB with parallel processing."""
import sys
import os
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def import_single_file(file_path: str) -> dict:
    """Worker function — imports a single file with its own DB and OCR instances."""
    from src.config import settings
    from src.db.connection import Database
    from src.ingestion.parser import DocumentParser
    from src.ingestion.cleaner import TextCleaner
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.orchestrator import IngestionOrchestrator

    db = Database(settings.db_path)
    db.initialize()
    orch = IngestionOrchestrator(
        db=db,
        parser=DocumentParser(enable_ocr=True),
        cleaner=TextCleaner(),
        chunker=SemanticChunker(
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
    )
    t0 = time.time()
    result = orch.ingest_file(file_path, source_status="main")
    elapsed = time.time() - t0
    db.close()
    result["file"] = os.path.basename(file_path)
    result["elapsed"] = elapsed
    return result


def main():
    import_dir = "参考资料"
    if not os.path.isdir(import_dir):
        print(f"Directory not found: {import_dir}")
        sys.exit(1)

    pdf_files = [
        os.path.join(import_dir, f)
        for f in sorted(os.listdir(import_dir))
        if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF files.")
    workers = min(3, len(pdf_files))
    print(f"Using {workers} parallel workers\n")

    # Process with streaming results
    total_chunks = 0
    total_quarantined = 0
    done = 0
    t_start = time.time()

    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(import_single_file, pdf_files):
            done += 1
            status = result.get("status", "?")
            name = result.get("file", "?")
            chunks = result.get("chunk_count", 0)
            quarantined = result.get("quarantined_count", 0)
            elapsed = result.get("elapsed", 0)
            msg = result.get("message", "")
            total_chunks += chunks
            total_quarantined += quarantined
            print(f"  [{done}/{len(pdf_files)}] [{status}] {name}: {chunks} chunks, {quarantined} quarantined ({elapsed:.0f}s)")
            if status == "failed":
                print(f"         Error: {msg}")

    elapsed = time.time() - t_start
    print(f"\nTotal: {total_chunks} chunks, {total_quarantined} quarantined in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("Done. Run POST /index/rebuild to build search indexes.")


if __name__ == "__main__":
    main()
