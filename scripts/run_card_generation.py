#!/usr/bin/env python3
"""Generate knowledge cards from enriched chunks using DeepSeek API.

Groups consecutive chunks within each book into triples,
generates one actionable nursing card per group.

Usage:
    python scripts/run_card_generation.py          # full run
    python scripts/run_card_generation.py --dry-run # test with 5 cards
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
from src.ingestion.tag_pool import validate_tags

PROGRESS_FILE = "data/card_progress.json"


def load_progress() -> set:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        return set(json.load(f))


def save_progress(done_groups: set) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done_groups), f)


def build_groups(chunks_by_book: dict) -> list[list[dict]]:
    """Group consecutive chunks into triples, sliding window of 3."""
    groups = []
    for book, chunks in chunks_by_book.items():
        chunks.sort(key=lambda c: (c.get("page_start", 0), c.get("page_end", 0)))
        for i in range(0, len(chunks) - 2, 2):  # stride=2 for overlap coverage
            group = chunks[i:i + 3]
            group_id = "+".join(c["id"][-8:] for c in group)
            groups.append({"id": group_id, "chunks": group, "book": book})
    return groups


async def main():
    dry_run = "--dry-run" in sys.argv
    resume = "--resume" in sys.argv

    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    # Get all enriched chunks grouped by book
    all_chunks = repo.get_all_active()
    chunks_by_book = {}
    for c in all_chunks:
        book = c.get("source_title", "unknown")
        chunks_by_book.setdefault(book, []).append(c)

    groups = build_groups(chunks_by_book)
    print(f"{len(all_chunks)} chunks → {len(groups)} card groups across {len(chunks_by_book)} books")

    if resume:
        done = load_progress()
        groups = [g for g in groups if g["id"] not in done]
        print(f"Resume: {len(groups)} groups remaining")
    else:
        done = set()

    if dry_run:
        groups = groups[:5]
        print("DRY RUN: 5 groups")

    if not groups:
        print("All cards generated!")
        db.close()
        return

    # LLM
    config = LLMConfig(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key,
        model=settings.llm_model, timeout=settings.llm_timeout,
    )
    enricher = LLMEnricher(LLMClient(config))

    print(f"Generating {len(groups)} cards via {settings.llm_model}")
    t_start = time.time()
    generated = 0
    errors = 0
    card_targets_dist = {}

    for i, group in enumerate(groups):
        try:
            card = await enricher.generate_card(group["chunks"])
            if card is None:
                errors += 1
                continue

            # Validate card targets
            card["card_targets"] = validate_tags(card_targets=card.get("card_targets", []))["card_targets"]

            # Insert as knowledge_card
            import uuid
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            card_id = f"ku_card_{uuid.uuid4().hex[:8]}"

            # Merge tags from parent chunks
            parent_tags = set()
            for pc in group["chunks"]:
                for tag in json.loads(pc.get("semantic_tags", "[]")):
                    parent_tags.add(tag)

            repo.insert({
                "id": card_id,
                "unit_type": "knowledge_card",
                "source_type": group["chunks"][0].get("source_type", "pdf_book"),
                "source_status": "main",
                "title": card["title"],
                "text": card["text"],
                "summary": "",
                "source_uri": group["chunks"][0].get("source_uri", ""),
                "source_title": group["book"],
                "source_citation": "",
                "page_start": group["chunks"][0].get("page_start"),
                "page_end": group["chunks"][-1].get("page_end"),
                "semantic_tags": json.dumps(list(parent_tags), ensure_ascii=False),
                "scenario_tags": "[]",
                "role_tags": json.dumps(["nurse"], ensure_ascii=False),
                "method_tags": "[]",
                "risk_levels": "[]",
                "card_targets": json.dumps(card["card_targets"], ensure_ascii=False),
                "contraindications": "[]",
                "quality_score": 0.9,
                "quality_flags": "[]",
                "review_status": "unreviewed",
                "parent_chunk_ids": json.dumps(card["parent_chunk_ids"], ensure_ascii=False),
                "parse_method": group["chunks"][0].get("parse_method", ""),
                "embedding_model": "",
                "created_at": now,
                "updated_at": now,
            })

            generated += 1
            done.add(group["id"])
            for ct in card["card_targets"]:
                card_targets_dist[ct] = card_targets_dist.get(ct, 0) + 1

            if generated % 20 == 0:
                elapsed = time.time() - t_start
                rate = generated / elapsed if elapsed > 0 else 0
                eta = (len(groups) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(groups)}] {generated} cards, {errors} errors "
                      f"({rate:.1f}/s, ETA {eta/60:.0f}min)")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on group {group['id'][:16]}: {e}")

        await asyncio.sleep(0.05)

    save_progress(done)
    elapsed = time.time() - t_start

    print(f"\n{'='*50}")
    print(f"Complete: {generated} cards, {errors} errors in {elapsed/60:.1f}min")
    print(f"Card targets: {card_targets_dist}")
    print(f"Progress saved to {PROGRESS_FILE}")

    # Final stats
    card_count = db.conn.execute(
        "SELECT COUNT(*) FROM knowledge_units WHERE unit_type = 'knowledge_card'"
    ).fetchone()[0]
    print(f"Total knowledge cards in DB: {card_count}")
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
