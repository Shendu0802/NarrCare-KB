"""Safety/contraindication rule-based forced recall."""

SAFETY_RULES = {
    "呼吸困难": ["呼吸困难", "呼吸窘迫", "气促", "喘息"],
    "自杀意念": ["自杀", "绝望", "不想活", "结束生命"],
    "疼痛危象": ["剧烈疼痛", "疼痛难忍", "疼痛控制"],
    "谵妄": ["谵妄", "意识模糊", "躁动"],
}


class SafetyRecaller:
    def __init__(self, repo):
        self.repo = repo

    def recall(self, contraindication_signals: list[str], risk_signals: list[str]) -> list[dict]:
        results = []
        seen = set()
        all_signals = contraindication_signals + risk_signals

        for signal in all_signals:
            rules = SAFETY_RULES.get(signal, [signal])
            for keyword in rules:
                for unit in self.repo.search_by_tags([keyword], "contraindications", 10):
                    if unit["id"] not in seen:
                        seen.add(unit["id"])
                        results.append({
                            "id": unit["id"], "score": 1.0, "source": "safety",
                            "safety_priority_boost": 0.2,
                        })
        return results
