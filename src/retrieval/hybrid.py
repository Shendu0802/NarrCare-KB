"""Multi-recall fusion with weighted scoring."""
from src.config import settings


class HybridFusion:
    def __init__(self):
        self.w_recall = settings.weight_recall
        self.w_metadata = settings.weight_metadata
        self.w_quality = settings.weight_quality
        self.w_source = settings.weight_source_status

    def fuse(
        self,
        dense_hits: list[dict], sparse_hits: list[dict],
        metadata_hits: list[dict], safety_hits: list[dict],
        knowledge_units: dict[str, dict],
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        source_weight_map = {"main": 0.05, "candidate": 0.025, "quarantined": 0.0}

        for hit in dense_hits + sparse_hits + metadata_hits + safety_hits:
            uid = hit["id"]
            if uid in merged:
                merged[uid]["score"] += hit.get("score", 0.0) * 0.5
                if hit.get("safety_priority_boost"):
                    merged[uid]["score"] += hit["safety_priority_boost"]
            else:
                unit = knowledge_units.get(uid, {})
                quality = unit.get("quality_score", 0.5) if isinstance(unit, dict) else 0.5
                src_status = unit.get("source_status", "candidate") if isinstance(unit, dict) else "candidate"

                score = (
                    hit.get("score", 0.0) * self.w_recall
                    + quality * self.w_quality
                    + source_weight_map.get(src_status, 0.0) * self.w_source
                )
                if hit.get("safety_priority_boost"):
                    score += hit["safety_priority_boost"]

                merged[uid] = {
                    "id": uid, "score": score,
                    "sources": [hit.get("source", "unknown")],
                    "safety_boost": hit.get("safety_priority_boost", 0.0),
                }

        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
