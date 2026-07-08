"""POST /retrieve -- full retrieval pipeline."""
from fastapi import APIRouter
from src.models.retrieval import RetrieveRequest
from src.models.evidence_bundle import EvidenceBundle, QueryAnalysis
from src.config import settings

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    from src.llm.client import LLMClient, LLMConfig
    from src.retrieval.query_understanding import QueryUnderstanding
    from src.db.connection import Database
    from src.db.repository import KnowledgeUnitRepository

    # Query understanding -- fallback to patient_text if LLM unavailable
    query_analysis = QueryAnalysis(rewritten_queries=[request.patient_text])
    try:
        llm_config = LLMConfig(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key,
            model=settings.llm_model, timeout=settings.llm_timeout,
        )
        llm_client = LLMClient(llm_config)
        qu_engine = QueryUnderstanding(llm_client)
        query_analysis = await qu_engine.analyze(
            patient_text=request.patient_text,
            family_text=request.family_text,
            risk_assessment=request.risk_assessment,
            dyadic_analysis=request.dyadic_analysis,
        )
    except Exception:
        pass  # Use fallback with original patient_text already set

    # Database
    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    # Load indexes and embedder
    from src.indexing.embedder import Embedder
    from src.indexing.vector_index import VectorIndex
    from src.indexing.text_index import TextIndex

    embedder = Embedder(model_path="api", device=settings.embedding_device)
    embedder.load()

    vec_index = VectorIndex(index_path=settings.faiss_index_path)
    vec_index.load()

    text_index = TextIndex(db)

    # Recall
    from src.retrieval.dense import DenseRecaller
    from src.retrieval.sparse import SparseRecaller
    from src.retrieval.metadata_recall import MetadataRecaller
    from src.retrieval.safety_recall import SafetyRecaller
    from src.retrieval.hybrid import HybridFusion

    dense_hits = DenseRecaller(embedder, vec_index).recall(query_analysis.rewritten_queries)
    sparse_hits = SparseRecaller(text_index).recall(query_analysis.rewritten_queries)
    metadata_hits = MetadataRecaller(repo).recall(query_analysis.scenario_tags, query_analysis.role_focus)
    safety_hits = SafetyRecaller(repo).recall(query_analysis.contraindication_signals, query_analysis.risk_signals)

    # Build unit lookup
    all_ids = set()
    for hits in [dense_hits, sparse_hits, metadata_hits, safety_hits]:
        for h in hits:
            all_ids.add(h["id"])

    unit_lookup = {}
    for uid in all_ids:
        unit = repo.get_by_id(uid)
        if unit:
            unit_lookup[uid] = unit

    # Fuse and bundle
    fused = HybridFusion().fuse(dense_hits, sparse_hits, metadata_hits, safety_hits, unit_lookup)

    from src.retrieval.bundler import Bundler
    bundle = Bundler().assemble(fused, unit_lookup, query_analysis,
                                top_k_cards=request.top_k_cards,
                                top_k_passages=request.top_k_passages)

    db.close()
    return {"data": bundle.model_dump(), "error": None}
