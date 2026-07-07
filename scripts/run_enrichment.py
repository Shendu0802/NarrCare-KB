#!/usr/bin/env python3
"""Run LLM enrichment against all chunks using DeepSeek API.

Generates summary + multidimensional tags for every chunk currently
missing enrichment data. Supports resume on interrupt.

Usage:
    python scripts/run_enrichment.py              # full run (5441 chunks)
    python scripts/run_enrichment.py --dry-run    # test with 5 chunks only
    python scripts/run_enrichment.py --resume     # skip already-enriched chunks
"""
import sys
import os
import json
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository
from src.llm.client import LLMClient, LLMConfig
from src.ingestion.enricher import LLMEnricher


PROGRESS_FILE = "data/enrichment_progress.json"


def load_progress() -> set:
    """Load IDs of already-enriched chunks."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        return set(json.load(f))


def save_progress(done_ids: set) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done_ids), f)


async def main():
    dry_run = "--dry-run" in sys.argv
    resume = "--resume" in sys.argv

    # Init
    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    # LLM client via DeepSeek
    config = LLMConfig(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
    llm_client = LLMClient(config)
    enricher = LLMEnricher(llm_client)

    # Get chunks to process
    all_chunks = repo.get_all_active()
    total = len(all_chunks)

    if resume:
        done = load_progress()
        chunks = [c for c in all_chunks if c["id"] not in done]
        print(f"Resuming: {len(done)}/{total} already done, {len(chunks)} remaining")
    else:
        done = set()
        chunks = all_chunks

    if dry_run:
        chunks = chunks[:5]
        print(f"DRY RUN: {len(chunks)} chunks")

    if not chunks:
        print("All chunks enriched!")
        db.close()
        return

    print(f"Enriching {len(chunks)} chunks via {settings.llm_model} @ {settings.llm_base_url}")
    t_start = time.time()
    enriched = 0
    errors = 0
    tag_counts = {}  # track tag distribution

    for i, chunk in enumerate(chunks):
        try:
            result = await enricher.enrich_chunk(chunk)

            repo.update_enrichment(
                ku_id=chunk["id"],
                summary=result.get("summary", ""),
                semantic_tags=result.get("semantic_tags", []),
                scenario_tags=result.get("scenario_tags", []),
                role_tags=result.get("role_tags", []),
                method_tags=result.get("method_tags", []),
                risk_levels=result.get("risk_levels", []),
                card_targets=result.get("card_targets", []),
            )

            enriched += 1
            done.add(chunk["id"])

            # Track tag distribution
            for tag in result.get("semantic_tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Progress every 50
            if enriched % 50 == 0:
                elapsed = time.time() - t_start
                rate = enriched / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(chunks)}] {enriched} enriched, {errors} errors "
                      f"({rate:.1f}/s, ETA {eta/60:.0f}min)")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on chunk {chunk['id'][:16]}: {e}")

        # Rate limiting
        await asyncio.sleep(0.05)

    # Final save
    save_progress(done)
    elapsed = time.time() - t_start

    print(f"\n{'='*50}")
    print(f"Complete: {enriched} enriched, {errors} errors in {elapsed/60:.1f}min")
    print(f"Rate: {enriched/elapsed:.1f} chunks/s")
    print(f"Top tags: {sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
    print(f"Progress saved to {PROGRESS_FILE}")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
