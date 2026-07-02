from src.models.knowledge_unit import KnowledgeUnit
from src.models.evidence_bundle import EvidenceItem, EvidenceBundle, QueryAnalysis, CardSources
from src.models.retrieval import RetrieveRequest
from src.models.ingestion import IngestFilesRequest, IngestSearchRequest


class TestKnowledgeUnit:
    def test_minimal_creation(self):
        ku = KnowledgeUnit(
            id="ku_pdf_book_a1b2c3d4",
            unit_type="semantic_chunk",
            source_type="pdf_book",
            text="临终患者常常经历存在性痛苦...",
            created_at="2026-07-01T00:00:00",
            updated_at="2026-07-01T00:00:00",
        )
        assert ku.id == "ku_pdf_book_a1b2c3d4"
        assert ku.source_status == "candidate"
        assert ku.quality_score == 0.0
        assert ku.semantic_tags == []

    def test_tag_serialization(self):
        ku = KnowledgeUnit(
            id="ku_test_1",
            unit_type="semantic_chunk",
            source_type="guideline",
            text="呼吸困难的非药物干预包括...",
            semantic_tags=["呼吸困难", "症状管理"],
            scenario_tags=["夜间焦虑"],
            role_tags=["nurse"],
            card_targets=["communication"],
            created_at="2026-07-01T00:00:00",
            updated_at="2026-07-01T00:00:00",
        )
        data = ku.model_dump()
        assert "呼吸困难" in data["semantic_tags"]
        assert "nurse" in data["role_tags"]

    def test_source_types(self):
        valid_types = [
            "pdf_book", "paper", "guideline", "case",
            "structured_library", "markdown", "web_reference",
        ]
        for t in valid_types:
            ku = KnowledgeUnit(
                id=f"ku_{t}_test",
                unit_type="semantic_chunk",
                source_type=t,
                text="test",
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
            )
            assert ku.source_type == t


class TestEvidenceBundle:
    def test_empty_bundle(self):
        bundle = EvidenceBundle(
            query_analysis=QueryAnalysis(),
            card_sources=CardSources(),
            supporting_passages=[],
            safety_and_boundary_evidence=[],
            candidate_evidence=[],
            excluded_evidence=[],
        )
        assert bundle.query_analysis.intents == []
        assert bundle.card_sources.mindfulness == []

    def test_bundle_with_items(self):
        item = EvidenceItem(
            id="ku_test_1",
            unit_type="semantic_chunk",
            source_type="guideline",
            source_status="main",
            title="呼吸困难干预",
            snippet="非药物干预包括...",
            summary="呼吸困难的非药物管理方法",
            score=0.85,
            source_citation="WHO Guideline 2024",
            source_uri="",
            semantic_tags=["呼吸困难"],
            scenario_tags=[],
            role_tags=["nurse"],
            method_tags=[],
            risk_levels=["high"],
            card_targets=["communication"],
            quality_score=0.9,
            parent_chunk_ids=[],
        )
        bundle = EvidenceBundle(
            query_analysis=QueryAnalysis(intents=["symptom_management"], scenario_tags=["呼吸困难"]),
            card_sources=CardSources(communication=[item]),
            supporting_passages=[item],
            safety_and_boundary_evidence=[],
            candidate_evidence=[],
            excluded_evidence=[],
        )
        assert len(bundle.supporting_passages) == 1
        assert len(bundle.card_sources.communication) == 1


class TestRetrieveRequest:
    def test_defaults(self):
        req = RetrieveRequest(patient_text="我感到害怕")
        assert req.user_role == "nurse"
        assert req.top_k_cards == 3
        assert req.top_k_passages == 5
        assert req.include_debug is False


class TestIngestRequests:
    def test_ingest_files_request(self):
        req = IngestFilesRequest(file_paths=["/data/book.pdf"])
        assert req.source_status == "main"

    def test_ingest_search_request(self):
        req = IngestSearchRequest(topic="death anxiety hospice")
        assert req.max_results == 20
        assert "pubmed" in req.source_priority
