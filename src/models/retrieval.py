"""Retrieval request/response schemas."""
from typing import Literal
from pydantic import BaseModel, Field
from src.models.evidence_bundle import EvidenceBundle


class RetrieveRequest(BaseModel):
    session_id: str = ""
    patient_text: str = ""
    family_text: str = ""
    user_role: Literal["patient", "family", "nurse"] = "nurse"
    risk_assessment: dict = Field(default_factory=dict)
    dyadic_analysis: dict = Field(default_factory=dict)
    top_k_cards: int = 3
    top_k_passages: int = 5
    include_debug: bool = False


class RetrieveResponse(BaseModel):
    data: EvidenceBundle
    error: str | None = None
