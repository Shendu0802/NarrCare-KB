"""Metadata/tag-based recall."""
from src.config import settings


class MetadataRecaller:
    def __init__(self, repo):
        self.repo = repo

    def recall(self, scenario_tags: list[str], role_focus: list[str], top_k: int | None = None) -> list[dict]:
        if top_k is None:
            top_k = settings.metadata_top_k
        results = []
        seen = set()

        if scenario_tags:
            for unit in self.repo.search_by_tags(scenario_tags, "scenario_tags", top_k):
                if unit["id"] not in seen:
                    seen.add(unit["id"])
                    results.append({"id": unit["id"], "score": 0.5, "source": "metadata"})

        if role_focus:
            for unit in self.repo.search_by_tags(role_focus, "role_tags", top_k):
                if unit["id"] not in seen:
                    seen.add(unit["id"])
                    results.append({"id": unit["id"], "score": 0.4, "source": "metadata"})

        return results
