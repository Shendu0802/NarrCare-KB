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
        if not returned_tags:
            return 0.0  # Known limitation: chunks have no LLM tags
        hits = set(returned_tags) & set(expected_tags)
        return len(hits) / len(expected_tags)

    @staticmethod
    def source_type_match(retrieved_items: list[dict], must_include_types: list[str]) -> float:
        """Fraction of expected source types found in results."""
        if not must_include_types:
            return 1.0
        found = set()
        for item in retrieved_items:
            st = item.get("source_type", "")
            if st in must_include_types:
                found.add(st)
        return len(found) / len(must_include_types)

    @staticmethod
    def flag_check(all_items: list[dict], forbidden_flags: list[str]) -> float:
        """1.0 if no forbidden flags appear in main results, decreasing otherwise."""
        if not forbidden_flags:
            return 1.0
        violations = 0
        for item in all_items:
            status = item.get("source_status", "")
            if status in forbidden_flags:
                violations += 1
            for qf in item.get("quality_flags", []):
                if qf in forbidden_flags:
                    violations += 1
        if violations == 0:
            return 1.0
        return max(0.0, 1.0 - violations / max(1, len(all_items)))

    @staticmethod
    def card_target_match(retrieved_items: list[dict], expected_cards: list[str]) -> float:
        """Fraction of expected card targets found in results."""
        if not expected_cards:
            return 1.0
        found = set()
        for item in retrieved_items:
            for ct in item.get("card_targets", []):
                if ct in expected_cards:
                    found.add(ct)
        return len(found) / len(expected_cards)
