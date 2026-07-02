"""Debug API endpoints: GET /debug/status, GET /debug/query."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.config import settings
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository

router = APIRouter(prefix="/debug", tags=["debug"])
templates = Jinja2Templates(directory="src/debug/templates")


@router.get("/status")
async def debug_status():
    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    counts = repo.count_by_status()
    type_counts = repo.count_by_type()

    docs = db.conn.execute("SELECT COUNT(*) as cnt FROM source_documents").fetchone()
    doc_count = docs["cnt"] if docs else 0

    src_dist = db.conn.execute(
        "SELECT source_type, COUNT(*) as cnt FROM knowledge_units GROUP BY source_type"
    ).fetchall()

    db.close()

    import torch
    return {
        "gpu_available": torch.cuda.is_available(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "index_version": "",
        "counts": {
            "documents": doc_count,
            "chunks": type_counts.get("semantic_chunk", 0),
            "cards": type_counts.get("knowledge_card", 0),
            "by_status": counts,
            "by_source_type": {r["source_type"]: r["cnt"] for r in src_dist},
        },
        "config": {
            "chunk_min_chars": settings.chunk_min_chars,
            "chunk_max_chars": settings.chunk_max_chars,
            "quality_threshold": settings.quality_main_threshold,
            "weights": {
                "rerank": settings.weight_rerank, "recall": settings.weight_recall,
                "metadata": settings.weight_metadata, "quality": settings.weight_quality,
                "source_status": settings.weight_source_status,
            },
        },
    }


@router.get("/query", response_class=HTMLResponse)
async def debug_query_page(request: Request):
    return templates.TemplateResponse("debug.html", {"request": request})
