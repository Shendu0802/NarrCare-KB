#!/usr/bin/env python3
"""Bake curated ground-truth chunk IDs into rag_queries.jsonl.

Reads data/eval/ground_truth_candidates.json (with manually marked relevant IDs)
and updates data/eval/rag_queries.jsonl with must_include_ids.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings


def main():
    gt_path = f"{settings.data_dir}/eval/ground_truth_candidates.json"
    queries_path = f"{settings.data_dir}/eval/rag_queries.jsonl"

    if not os.path.exists(gt_path):
        print(f"ERROR: {gt_path} not found. Run scripts/discover_ground_truth.py first.")
        sys.exit(1)

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    updated = 0
    for gt in ground_truth:
        qi = gt["query_idx"]
        # Collect IDs marked as relevant (relevance_note == "relevant")
        relevant_ids = []
        for c in gt.get("candidates", []):
            note = c.get("relevance_note", "")
            if note == "relevant":
                relevant_ids.append(c["id"])

        if relevant_ids:
            queries[qi]["must_include_ids"] = relevant_ids
            updated += 1
            print(f"  Query {qi}: {len(relevant_ids)} IDs → {gt['query'][:60]}...")
        else:
            print(f"  Query {qi}: 0 IDs (no candidates marked 'relevant') → {gt['query'][:60]}...")

    # Write updated queries
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nUpdated {updated}/{len(queries)} queries with must_include_ids")
    print(f"Saved to {queries_path}")
    print("Next: run POST /eval/run to execute evaluation")


if __name__ == "__main__":
    main()
