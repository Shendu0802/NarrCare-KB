"""Assembles scored hits into layered EvidenceBundle."""
from src.models.evidence_bundle import EvidenceBundle, EvidenceItem, CardSources


class Bundler:
    def assemble(
        self, fused_hits: list[dict], knowledge_units: dict[str, dict],
        query_analysis, top_k_cards: int = 3, top_k_passages: int = 5,
    ) -> EvidenceBundle:
        card_bins = {"mindfulness": [], "healing": [], "communication": [], "personalized": []}
        supporting = []
        safety = []
        candidate_evidence = []
        excluded = []

        for hit in fused_hits:
            unit = knowledge_units.get(hit["id"])
            if unit is None:
                continue

            item = EvidenceItem(
                id=hit["id"],
                unit_type=unit.get("unit_type", "semantic_chunk"),
                source_type=unit.get("source_type", ""),
                source_status=unit.get("source_status", ""),
                title=unit.get("title", ""),
                snippet=unit.get("text", "")[:300],
                summary=unit.get("summary", ""),
                score=hit["score"],
                source_citation=unit.get("source_citation", ""),
                source_uri=unit.get("source_uri", ""),
                page_start=unit.get("page_start"),
                page_end=unit.get("page_end"),
                semantic_tags=_parse_json(unit.get("semantic_tags", "[]")),
                scenario_tags=_parse_json(unit.get("scenario_tags", "[]")),
                role_tags=_parse_json(unit.get("role_tags", "[]")),
                method_tags=_parse_json(unit.get("method_tags", "[]")),
                risk_levels=_parse_json(unit.get("risk_levels", "[]")),
                card_targets=_parse_json(unit.get("card_targets", "[]")),
                quality_score=unit.get("quality_score", 0.0),
                review_status=unit.get("review_status", ""),
                parent_chunk_ids=_parse_json(unit.get("parent_chunk_ids", "[]")),
            )

            status = unit.get("source_status", "")
            if status == "quarantined":
                excluded.append(item)
            elif hit.get("safety_boost", 0) > 0:
                safety.append(item)
            elif status == "candidate":
                candidate_evidence.append(item)
            else:
                supporting.append(item)

            for ct in item.card_targets:
                if ct in card_bins and len(card_bins[ct]) < top_k_cards:
                    card_bins[ct].append(item)

        return EvidenceBundle(
            query_analysis=query_analysis,
            card_sources=CardSources(
                mindfulness=card_bins["mindfulness"][:top_k_cards],
                healing=card_bins["healing"][:top_k_cards],
                communication=card_bins["communication"][:top_k_cards],
                personalized=card_bins["personalized"][:top_k_cards],
            ),
            supporting_passages=supporting[:top_k_passages],
            safety_and_boundary_evidence=safety,
            candidate_evidence=candidate_evidence,
            excluded_evidence=excluded,
        )


def _parse_json(val):
    import json
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []
