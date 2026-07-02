"""Ingestion request/response schemas."""
from typing import Optional
from pydantic import BaseModel, Field


class IngestFilesRequest(BaseModel):
    file_paths: list[str]
    source_type: Optional[str] = None
    source_status: str = "main"


class IngestFilesResponse(BaseModel):
    task_id: str
    status: str
    results: list[dict] = Field(default_factory=list)


class IngestSearchRequest(BaseModel):
    topic: str
    keywords: list[str] = Field(default_factory=list)
    source_priority: list[str] = Field(
        default_factory=lambda: ["pubmed", "pmc", "who", "guideline"]
    )
    max_results: int = 20


class IngestSearchResponse(BaseModel):
    task_id: str
    candidates: list[dict] = Field(default_factory=list)
    status: str = "pending"
