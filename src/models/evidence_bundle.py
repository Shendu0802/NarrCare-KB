"""EvidenceBundle and supporting schemas."""
from typing import Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    id: str
    unit_type: str = ""
    source_type: str = ""
    source_status: str = ""
    title: str = ""
    snippet: str = ""
    summary: str = ""
    score: float = 0.0
    source_citation: str = ""
    source_uri: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    semantic_tags: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    role_tags: list[str] = Field(default_factory=list)
    method_tags: list[str] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)
    card_targets: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    review_status: str = ""
    parent_chunk_ids: list[str] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    intents: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    role_focus: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)
    contraindication_signals: list[str] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)


class CardSources(BaseModel):
    mindfulness: list[EvidenceItem] = Field(default_factory=list)
    healing: list[EvidenceItem] = Field(default_factory=list)
    communication: list[EvidenceItem] = Field(default_factory=list)
    personalized: list[EvidenceItem] = Field(default_factory=list)


class RetrievalDebug(BaseModel):
    dense_hits: list = Field(default_factory=list)
    sparse_hits: list = Field(default_factory=list)
    metadata_hits: list = Field(default_factory=list)
    reranked_hits: list = Field(default_factory=list)
    model_versions: dict = Field(default_factory=dict)
    index_version: str = ""


class EvidenceBundle(BaseModel):
    query_analysis: QueryAnalysis
    card_sources: CardSources
    supporting_passages: list[EvidenceItem] = Field(default_factory=list)
    safety_and_boundary_evidence: list[EvidenceItem] = Field(default_factory=list)
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)
    excluded_evidence: list[EvidenceItem] = Field(default_factory=list)
    retrieval_debug: Optional[RetrievalDebug] = None
