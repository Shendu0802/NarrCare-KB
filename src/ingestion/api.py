"""Ingestion API endpoints: POST /ingest/files, POST /ingest/search."""
from fastapi import APIRouter
from src.models.ingestion import (
    IngestFilesRequest, IngestFilesResponse,
    IngestSearchRequest, IngestSearchResponse,
)
from src.config import settings
from src.errors import KBException, KBErrorCode

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from src.db.connection import Database
        from src.ingestion.parser import DocumentParser
        from src.ingestion.cleaner import TextCleaner
        from src.ingestion.chunker import SemanticChunker
        from src.ingestion.orchestrator import IngestionOrchestrator

        db = Database(settings.db_path)
        db.initialize()
        _orchestrator = IngestionOrchestrator(
            db=db, parser=DocumentParser(), cleaner=TextCleaner(),
            chunker=SemanticChunker(
                min_chars=settings.chunk_min_chars,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ),
        )
    return _orchestrator


@router.post("/files", response_model=IngestFilesResponse)
async def ingest_files(request: IngestFilesRequest):
    import uuid
    orchestrator = get_orchestrator()
    try:
        results = orchestrator.ingest_batch(
            file_paths=request.file_paths,
            source_type=request.source_type,
            source_status=request.source_status,
        )
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        return IngestFilesResponse(task_id=task_id, status="done", results=results)
    except Exception as e:
        raise KBException(error_code=KBErrorCode.KB_INGESTION_FAILED, detail=str(e), http_status=500)


@router.post("/search", response_model=IngestSearchResponse)
async def ingest_search(request: IngestSearchRequest):
    import uuid
    return IngestSearchResponse(
        task_id=f"task_{uuid.uuid4().hex[:8]}", candidates=[], status="not_implemented",
    )
