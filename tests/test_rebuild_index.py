"""Rebuild FAISS index using Qwen DashScope embeddings API.

Usage:
    python tests/test_rebuild_index.py

This replaces TF-IDF vectors with semantic embeddings from Qwen text-embedding-v3.
Requires: valid KB_LLM_API_KEY in .env or environment.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository
from src.indexing.embedder import Embedder
from src.indexing.vector_index import VectorIndex
from src.indexing.text_index import TextIndex


def main():
    # Ensure API key is available
    if not settings.llm_api_key:
        print("ERROR: KB_LLM_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    active = repo.get_all_active()
    if not active:
        print("ERROR: No knowledge units in database. Run import first.")
        db.close()
        sys.exit(1)

    texts = [u['text'] for u in active]
    ids = [u['id'] for u in active]
    print(f"Building index for {len(active)} chunks...")

    # FTS5
    TextIndex(db).rebuild()
    print("FTS5 rebuilt")

    # API embeddings
    emb = Embedder(model_path="api")
    if not emb.load():
        print("ERROR: Failed to initialize API embedder")
        db.close()
        sys.exit(1)
    print(f"Embedding via {settings.llm_base_url} (model={emb._api_model})")

    t0 = time.time()
    embs = emb.encode(texts, batch_size=10)
    elapsed = time.time() - t0
    print(f"Encoded {len(active)} chunks in {elapsed:.0f}s "
          f"({len(active)/elapsed:.0f} chunks/s)")
    print(f"Shape: {embs.shape}")

    # Build FAISS
    vi = VectorIndex(dim=embs.shape[1], index_path=settings.faiss_index_path)
    vi.build(ids, embs)
    vi.save()
    print(f"FAISS index saved: {vi.size} vectors, {os.path.getsize(settings.faiss_index_path)/1024/1024:.1f}MB")

    db.close()
    print("=== Rebuild complete ===")
    print(f"Run: uvicorn src.main:app --host 0.0.0.0 --port {settings.port}")


if __name__ == "__main__":
    main()
