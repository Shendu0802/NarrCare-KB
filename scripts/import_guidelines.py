#!/usr/bin/env python3
"""Import guidelines/consensus/standards from the NarrCare guidelines folder.

Reads 00_download_manifest/download_manifest.csv to determine priority and
source_type. Skips technical references (not clinical content).
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db.connection import Database
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker
from src.ingestion.orchestrator import IngestionOrchestrator

BASE_DIR = "参考资料/NarrCare_相关指南下载"
MANIFEST = os.path.join(BASE_DIR, "00_download_manifest", "download_manifest.csv")

# Source type mapping from manifest → our schema
SOURCE_TYPE_MAP = {
    "guideline": "guideline",
    "consensus": "guideline",
    "standard": "guideline",
    "patient_guide": "guideline",
    "patient_education": "guideline",
    "patient_guideline": "guideline",
    "guideline_update": "guideline",
    "international_guideline": "guideline",
    "patient_professional_summary": "guideline",
    "guideline_insight": "guideline",
    "book_guideline": "guideline",
    "article": "paper",
    "method_paper": None,  # skip
    "technical_reference": None,
    "ai_governance": None,
    "reporting_guideline": None,
    "historical_guideline": "guideline",
}

# Priority → source_status
PRIORITY_STATUS = {"P0": "main", "P1": "main", "P2": "candidate"}

def main():
    if not os.path.exists(MANIFEST):
        print(f"ERROR: Manifest not found: {MANIFEST}")
        sys.exit(1)

    # Parse manifest
    items = []
    with open(MANIFEST, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)

    # Build filename → full_path index by walking the directory tree
    file_index = {}
    for root, dirs, files in os.walk(BASE_DIR):
        for fname in files:
            if fname.startswith("."):
                continue
            full = os.path.join(root, fname)
            # Index by both full filename and basename
            file_index[fname] = full
            # Also index without the path prefix for cross-reference
            file_index[os.path.basename(fname)] = full

    # Filter and resolve paths
    to_import = []
    skipped = 0
    for item in items:
        src_type = SOURCE_TYPE_MAP.get(item.get("source_type_for_RAG", "").strip())
        if src_type is None:
            skipped += 1
            continue
        local = item.get("local_path", "").strip()
        if not local:
            skipped += 1
            continue

        # Extract filename from Windows path
        fname = local.replace("\\", "/").split("/")[-1]
        # Remove size info like " | E:\..." if present
        if " | " in fname:
            fname = fname.split(" | ")[0]

        # Find in index
        full_path = file_index.get(fname)
        if full_path is None:
            # Try partial match
            for k, v in file_index.items():
                if fname[:60] in k or k in fname[:60]:
                    full_path = v
                    break
        if full_path is None:
            print(f"  SKIP: {item['item_id']} — {fname[:50]}...")
            skipped += 1
            continue

        to_import.append({
            "item_id": item["item_id"],
            "title": item.get("normalized_title", item["title"]),
            "path": full_path,
            "source_type": src_type,
            "priority": item.get("priority", "P2").strip(),
            "source_status": PRIORITY_STATUS.get(item.get("priority", "P2").strip(), "candidate"),
            "ext": os.path.splitext(full_path)[1].lower(),
        })

    print(f"Manifest: {len(items)} items")
    print(f"To import: {len(to_import)} | Skipped: {skipped} (non-clinical or missing)")
    print()

    # Init pipeline
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

    total_chunks = 0
    for i, item in enumerate(to_import):
        print(f"[{i+1}/{len(to_import)}] [{item['priority']}] {item['title'][:60]}... ({item['ext']})")
        result = orch.ingest_file(
            file_path=item["path"],
            source_type=item["source_type"],
            source_status=item["source_status"],
        )
        status = result["status"]
        chunks = result.get("chunk_count", 0)
        total_chunks += chunks
        if status == "skipped":
            print(f"  → {status}")
        elif status == "failed":
            print(f"  → {status}: {result.get('message', '')[:80]}")
        else:
            print(f"  → {status}: {chunks} chunks, {result.get('quarantined_count', 0)} quarantined")

    db.close()
    print(f"\nTotal new chunks: {total_chunks}")
    print("Done. Run enrichment and index rebuild if needed.")


if __name__ == "__main__":
    main()
