"""Core KnowledgeUnit schema."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

SourceType = Literal[
    "pdf_book", "paper", "guideline", "case",
    "structured_library", "markdown", "web_reference",
]
SourceStatus = Literal["main", "candidate", "quarantined"]
UnitType = Literal["semantic_chunk", "knowledge_card"]
ReviewStatus = Literal["unreviewed", "approved", "rejected"]
CardTarget = Literal["mindfulness", "healing", "communication", "personalized"]


class KnowledgeUnit(BaseModel):
    id: str
    unit_type: UnitType
    source_type: SourceType
    source_status: SourceStatus = "candidate"
    title: str = ""
    text: str
    summary: str = ""
    source_uri: str = ""
    source_title: str = ""
    source_citation: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    semantic_tags: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    role_tags: list[str] = Field(default_factory=list)
    method_tags: list[str] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)
    card_targets: list[CardTarget] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    review_status: ReviewStatus = "unreviewed"
    parent_chunk_ids: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    created_at: str
    updated_at: str
