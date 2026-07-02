"""NarrCare-KB FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.errors import KBException
from src.health.api import router as health_router
from src.ingestion.api import router as ingestion_router
from src.retrieval.api import router as retrieval_router
from src.indexing.api import router as indexing_router
from src.evaluation.api import router as eval_router
from src.debug.api import router as debug_router

app = FastAPI(
    title="NarrCare-KB",
    description="Independent Knowledge Base Service for NarrCare",
    version="0.1.0",
)

# Register routers
app.include_router(health_router)
app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(indexing_router)
app.include_router(eval_router)
app.include_router(debug_router)


@app.exception_handler(KBException)
async def kb_exception_handler(request: Request, exc: KBException):
    """Convert KBException to structured JSON error response."""
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.error_code, "detail": exc.detail},
    )
