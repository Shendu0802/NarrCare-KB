"""POST /eval/run endpoint."""
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter
from src.config import settings
from src.evaluation.metrics import EvalMetrics

router = APIRouter(prefix="/eval", tags=["evaluation"])


async def _retrieve_direct(query_text: str) -> dict:
    """Run retrieval with raw query — no LLM rewriting (deterministic eval)."""
    from src.models.evidence_bundle import QueryAnalysis
    from src.db.connection import Database
    from src.db.repository import KnowledgeUnitRepository
    from src.indexing.embedder import Embedder
    from src.indexing.vector_index import VectorIndex
    from src.indexing.text_index import TextIndex
    from src.retrieval.dense import DenseRecaller
    from src.retrieval.metadata_recall import MetadataRecaller
    from src.retrieval.safety_recall import SafetyRecaller
    from src.retrieval.hybrid import HybridFusion
    from src.retrieval.bundler import Bundler

    qa = QueryAnalysis(rewritten_queries=[query_text])

    db = Database(settings.db_path); db.initialize()
    repo = KnowledgeUnitRepository(db)

    emb = Embedder(model_path="api"); emb.load()
    vi = VectorIndex(index_path=settings.faiss_index_path); vi.load()

    dense_hits = DenseRecaller(emb, vi).recall([query_text])
    metadata_hits = MetadataRecaller(repo).recall([], [])
    safety_hits = SafetyRecaller(repo).recall([], [])

    all_ids = set()
    for hits in [dense_hits, metadata_hits, safety_hits]:
        for h in hits:
            all_ids.add(h["id"])

    unit_lookup = {uid: repo.get_by_id(uid) for uid in all_ids if repo.get_by_id(uid)}

    fused = HybridFusion().fuse(dense_hits, [], metadata_hits, safety_hits, unit_lookup)
    bundle = Bundler().assemble(fused, unit_lookup, qa)
    db.close()
    return bundle.model_dump()


@router.post("/run")
async def run_eval():
    eval_path = f"{settings.data_dir}/eval/rag_queries.jsonl"

    queries = []
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
    except FileNotFoundError:
        return {"status": "error", "message": f"Eval file not found: {eval_path}"}

    if not queries:
        return {"status": "error", "message": "No eval queries found"}

    results = []
    scenario_stats = {}  # aggregate by scenario

    for q in queries:
        from src.models.retrieval import RetrieveRequest
        from src.retrieval.api import retrieve

        query_text = q.get("query", "")
        try:
            bundle = await _retrieve_direct(query_text)
            all_items = (
                bundle["supporting_passages"]
                + bundle["card_sources"]["mindfulness"]
                + bundle["card_sources"]["healing"]
                + bundle["card_sources"]["communication"]
                + bundle["card_sources"]["personalized"]
            )
            retrieved_ids = [item["id"] for item in all_items]
            all_tags = []
            for item in all_items:
                all_tags.extend(item.get("semantic_tags", []))
                all_tags.extend(item.get("scenario_tags", []))

            # Core metrics
            recall = EvalMetrics.recall_at_k(retrieved_ids, q.get("must_include_ids", []))
            safety = EvalMetrics.safety_hit(
                bundle.get("safety_and_boundary_evidence", []), q.get("risk_level", "")
            )
            noise = EvalMetrics.noise_rate(
                bundle.get("excluded_evidence", []), len(all_items)
            )
            tag_score = EvalMetrics.tag_match(all_tags, q.get("expected_tags", []))

            # New metrics
            st_match = EvalMetrics.source_type_match(
                all_items, q.get("must_include_source_types", [])
            )
            flg_check = EvalMetrics.flag_check(
                all_items, q.get("must_not_include_flags", [])
            )
            ct_match = EvalMetrics.card_target_match(
                all_items, q.get("must_include_card_targets", [])
            )

            result = {
                "query": q["query"][:60],
                "scenario": q.get("scenario", ""),
                "risk_level": q.get("risk_level", ""),
                "recall@10": recall,
                "safety_hit": safety,
                "noise_rate": noise,
                "tag_match": tag_score,
                "source_type_match": st_match,
                "flag_check": flg_check,
                "card_target_match": ct_match,
            }
            results.append(result)

            # Per-scenario aggregation
            sc = q.get("scenario", "unknown")
            if sc not in scenario_stats:
                scenario_stats[sc] = {
                    "count": 0, "recall": 0, "safety": 0,
                    "noise": 0, "source_type": 0, "flag": 0,
                }
            ss = scenario_stats[sc]
            ss["count"] += 1
            ss["recall"] += recall
            ss["safety"] += 1 if safety else 0
            ss["noise"] += noise
            ss["source_type"] += st_match
            ss["flag"] += flg_check

        except Exception as e:
            results.append({"query": q["query"][:60], "scenario": q.get("scenario", ""), "error": str(e)})

    n = len(results)
    if n == 0:
        return {"status": "error", "message": "All queries failed"}

    avg_recall = sum(r.get("recall@10", 0) for r in results) / n
    avg_safety = sum(1 for r in results if r.get("safety_hit", False)) / n
    avg_noise = sum(r.get("noise_rate", 0) for r in results) / n
    avg_tag = sum(r.get("tag_match", 0) for r in results) / n
    avg_source_type = sum(r.get("source_type_match", 0) for r in results) / n
    avg_flag = sum(r.get("flag_check", 0) for r in results) / n
    avg_card = sum(r.get("card_target_match", 0) for r in results) / n

    # Per-scenario averages
    for sc, ss in scenario_stats.items():
        c = ss["count"]
        ss["avg_recall"] = round(ss["recall"] / c, 4)
        ss["avg_safety"] = round(ss["safety"] / c, 4)
        ss["avg_source_type"] = round(ss["source_type"] / c, 4)
        del ss["recall"], ss["safety"], ss["source_type"], ss["noise"], ss["flag"]

    from src.db.connection import Database
    db = Database(settings.db_path)
    db.initialize()
    now = datetime.now(timezone.utc).isoformat()
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    db.conn.execute(
        """INSERT INTO eval_runs (id, run_at, total_queries, recall_at_10, safety_hit_rate, noise_rate, avg_usability, details_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, now, n, avg_recall, avg_safety, avg_noise, avg_tag,
         json.dumps(results, ensure_ascii=False)),
    )
    db.conn.commit()
    db.close()

    # Generate report
    eval_summary = {
        "run_id": run_id, "total_queries": n,
        "recall_at_10": round(avg_recall, 4),
        "safety_hit_rate": round(avg_safety, 4),
        "noise_rate": round(avg_noise, 4),
        "avg_tag_match": round(avg_tag, 4),
        "avg_source_type_match": round(avg_source_type, 4),
        "avg_flag_check": round(avg_flag, 4),
        "avg_card_target_match": round(avg_card, 4),
        "by_scenario": scenario_stats,
        "details": results,
    }

    from src.evaluation.report import generate_console_summary, generate_report
    generate_console_summary(eval_summary)
    report_path = generate_report(eval_summary)
    eval_summary["report_path"] = report_path

    return eval_summary
