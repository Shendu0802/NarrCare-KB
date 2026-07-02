"""GET /health endpoint."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    gpu_available: bool
    embedding_model: str
    reranker_model: str
    index_loaded: bool
    index_version: str
    document_count: int
    chunk_count: int
    card_count: int


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health status."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    from src.config import settings

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        index_loaded=False,
        index_version="",
        document_count=0,
        chunk_count=0,
        card_count=0,
    )
