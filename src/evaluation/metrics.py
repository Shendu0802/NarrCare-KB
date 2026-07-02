"""Evaluation metrics for retrieval quality."""


class EvalMetrics:
    @staticmethod
    def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 10) -> float:
        if not expected_ids:
            return 1.0
        top_k = set(retrieved_ids[:k])
        hits = top_k & set(expected_ids)
        return len(hits) / len(expected_ids)

    @staticmethod
    def safety_hit(safety_evidence: list, risk_level: str) -> bool:
        if risk_level in ("high", "medium"):
            return len(safety_evidence) > 0
        return True

    @staticmethod
    def noise_rate(excluded: list, total_returned: int) -> float:
        if total_returned == 0:
            return 0.0
        quarantined = sum(1 for e in excluded if e.get("source_status") == "quarantined")
        return quarantined / total_returned

    @staticmethod
    def tag_match(returned_tags: list[str], expected_tags: list[str]) -> float:
        if not expected_tags:
            return 1.0
        hits = set(returned_tags) & set(expected_tags)
        return len(hits) / len(expected_tags)
