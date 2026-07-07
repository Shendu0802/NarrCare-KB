#!/usr/bin/env python3
"""Discover ground-truth chunk IDs for evaluation queries.

For each query in rag_queries.jsonl:
1. Dense recall via Qwen embedding + FAISS (top-20)
2. FTS5 keyword search with scenario-specific Chinese keywords
3. Merge, deduplicate, fetch full chunk text
4. Output candidates to data/eval/ground_truth_candidates.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository
from src.indexing.embedder import Embedder
from src.indexing.vector_index import VectorIndex
from src.indexing.text_index import TextIndex
from src.retrieval.dense import DenseRecaller
from src.retrieval.sparse import SparseRecaller


# Scenario-specific FTS5 keywords for targeted sparse recall
SCENARIO_KEYWORDS = {
    "死亡焦虑": "死亡 OR 恐惧 OR 焦虑 OR 存在 OR 临终 OR 生命意义",
    "夜间恐惧": "夜间 OR 夜晚 OR 失眠 OR 噩梦 OR 睡不着 OR 黑暗 OR 惊醒",
    "家属否认": "家属 OR 隐瞒 OR 否认 OR 不接受 OR 告知 OR 病情",
    "照护者负担": "照护 OR 照顾 OR 负担 OR 倦怠 OR 疲劳 OR 耗竭 OR 自我关怀",
    "未竟事务": "未竟 OR 遗憾 OR 未完成 OR 遗愿 OR 和解 OR 告别 OR 回忆录",
    "告别沟通": "告别 OR 沟通 OR 对话 OR 道别 OR 最后 OR 再见 OR 对不起 OR 谢谢",
    "治疗拒绝": "拒绝治疗 OR 不治 OR 放弃 OR 化疗 OR 鼻饲 OR 插管 OR 回家",
    "预后不确定": "预后 OR 不确定 OR 几个月 OR 时好时坏 OR 反复 OR 希望 OR 失望",
    "呼吸困难": "呼吸 OR 喘息 OR 窒息 OR 氧气 OR 吗啡 OR 喘不上 OR 气促",
    "高风险绝望": "自杀 OR 绝望 OR 想死 OR 不想活 OR 结束 OR 绝食 OR 抑郁",
}


def main():
    queries_path = f"{settings.data_dir}/eval/rag_queries.jsonl"
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries from {queries_path}")

    # Init
    db = Database(settings.db_path); db.initialize()
    repo = KnowledgeUnitRepository(db)

    emb = Embedder("api"); emb.load()
    vi = VectorIndex(index_path=settings.faiss_index_path); vi.load()
    ti = TextIndex(db)

    dense_recaller = DenseRecaller(emb, vi)
    sparse_recaller = SparseRecaller(ti)

    all_candidates = []
    t0 = time.time()

    for qi, q in enumerate(queries):
        query_text = q["query"]
        scenario = q.get("scenario", "")
        print(f"\n[{qi+1}/{len(queries)}] {scenario}: {query_text[:60]}...")

        # Dense recall
        q_vec = emb.encode_query(query_text)
        dense_hits = dense_recaller.recall([query_text], top_k=20)

        # Sparse recall with scenario keywords
        kw = SCENARIO_KEYWORDS.get(scenario, query_text)
        sparse_hits = sparse_recaller.recall([kw], top_k=15)

        # Merge by ID, keeping best score
        merged = {}
        for h in dense_hits:
            merged[h["id"]] = {"id": h["id"], "score": h.get("score", 0), "sources": ["dense"]}
        for h in sparse_hits:
            sid = h["id"]
            if sid in merged:
                merged[sid]["score"] = max(merged[sid]["score"], float(h.get("score", 0)))
                merged[sid]["sources"].append("sparse")
            else:
                merged[sid] = {"id": sid, "score": float(h.get("score", 0)), "sources": ["sparse"]}

        # Sort and take top 15
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:15]

        # Fetch full text previews
        candidates = []
        for item in ranked:
            unit = repo.get_by_id(item["id"])
            if unit:
                candidates.append({
                    "id": item["id"],
                    "score": round(item["score"], 4),
                    "sources": "+".join(item["sources"]),
                    "source_title": unit.get("source_title", ""),
                    "page_start": unit.get("page_start"),
                    "page_end": unit.get("page_end"),
                    "text_preview": (unit.get("text", "") or "")[:300],
                })

        all_candidates.append({
            "query_idx": qi,
            "query": query_text,
            "scenario": scenario,
            "risk_level": q.get("risk_level", ""),
            "expected_tags": q.get("expected_tags", []),
            "candidates": candidates,
        })

        print(f"  dense={len(dense_hits)} sparse={len(sparse_hits)} merged={len(candidates)}")
        if candidates:
            print(f"  top: [{candidates[0]['score']:.3f}] {candidates[0]['source_title']} p{candidates[0]['page_start']}")

    db.close()

    # Write output
    out_path = f"{settings.data_dir}/eval/ground_truth_candidates.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\nSaved {len(all_candidates)} query results to {out_path}")
    print(f"Time: {elapsed:.0f}s ({elapsed/len(queries):.1f}s/query)")

    # Stats
    avg_candidates = sum(len(qc["candidates"]) for qc in all_candidates) / len(all_candidates)
    print(f"Avg candidates per query: {avg_candidates:.1f}")
    print("\nNext: review candidates and run scripts/enrich_queries.py to set must_include_ids")


if __name__ == "__main__":
    main()
