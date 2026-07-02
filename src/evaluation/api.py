"""POST /eval/run endpoint."""
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter
from src.config import settings
from src.evaluation.metrics import EvalMetrics

router = APIRouter(prefix="/eval", tags=["evaluation"])


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
    for q in queries:
        from src.models.retrieval import RetrieveRequest
        from src.retrieval.api import retrieve

        request = RetrieveRequest(patient_text=q.get("query", ""), user_role="nurse")
        try:
            response = await retrieve(request)
            bundle = response["data"]
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

            recall = EvalMetrics.recall_at_k(retrieved_ids, q.get("must_include_ids", []))
            safety = EvalMetrics.safety_hit(bundle.get("safety_and_boundary_evidence", []), q.get("risk_level", ""))
            noise = EvalMetrics.noise_rate(bundle.get("excluded_evidence", []), len(all_items))
            tag_score = EvalMetrics.tag_match(all_tags, q.get("expected_tags", []))

            results.append({
                "query": q["query"][:60], "recall@10": recall,
                "safety_hit": safety, "noise_rate": noise, "tag_match": tag_score,
            })
        except Exception as e:
            results.append({"query": q["query"][:60], "error": str(e)})

    n = len(results)
    avg_recall = sum(r.get("recall@10", 0) for r in results) / n if n else 0
    avg_safety = sum(1 for r in results if r.get("safety_hit", False)) / n if n else 0
    avg_noise = sum(r.get("noise_rate", 0) for r in results) / n if n else 0
    avg_tag = sum(r.get("tag_match", 0) for r in results) / n if n else 0

    from src.db.connection import Database
    db = Database(settings.db_path)
    db.initialize()
    now = datetime.now(timezone.utc).isoformat()
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    db.conn.execute(
        """INSERT INTO eval_runs (id, run_at, total_queries, recall_at_10, safety_hit_rate, noise_rate, avg_usability, details_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, now, n, avg_recall, avg_safety, avg_noise, avg_tag, json.dumps(results, ensure_ascii=False)),
    )
    db.conn.commit()
    db.close()

    return {
        "run_id": run_id, "total_queries": n,
        "recall_at_10": round(avg_recall, 4),
        "safety_hit_rate": round(avg_safety, 4),
        "noise_rate": round(avg_noise, 4),
        "avg_tag_match": round(avg_tag, 4),
        "details": results,
    }
