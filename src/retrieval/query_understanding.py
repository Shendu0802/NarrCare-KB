"""LLM-based query understanding: intent analysis and query rewriting."""
from src.models.evidence_bundle import QueryAnalysis

QU_SYSTEM_PROMPT = """You are a clinical query analyzer for a hospice care knowledge base.
Analyze the user's input and generate:

1. intents: What does the user need? (e.g. 死亡焦虑缓解, 家属沟通建议)
2. scenario_tags: What care scenarios are involved?
3. role_focus: Which roles need attention? (patient, family, nurse)
4. risk_signals: Any risk indicators in the text?
5. contraindication_signals: Any contraindication concerns?
6. rewritten_queries: At least 4 query variants:
   - Original expression preserved
   - Scenario-oriented query
   - Nursing intervention query
   - Safety/contraindication query

Return valid JSON only."""


class QueryUnderstanding:
    """Analyzes raw user input and generates structured query understanding."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze(
        self,
        patient_text: str,
        family_text: str = "",
        risk_assessment: dict | None = None,
        dyadic_analysis: dict | None = None,
    ) -> QueryAnalysis:
        user_text = f"Patient: {patient_text}\nFamily: {family_text}"
        if risk_assessment:
            user_text += f"\nRisk: {risk_assessment}"
        if dyadic_analysis:
            user_text += f"\nDyadic: {dyadic_analysis}"

        messages = self.llm.build_messages(system=QU_SYSTEM_PROMPT, user=user_text)

        try:
            result = await self.llm.chat_with_json(messages, temperature=0.3, max_tokens=1024)
        except Exception:
            result = {}

        return QueryAnalysis(
            intents=result.get("intents", []),
            scenario_tags=result.get("scenario_tags", []),
            role_focus=result.get("role_focus", []),
            risk_signals=result.get("risk_signals", []),
            contraindication_signals=result.get("contraindication_signals", []),
            rewritten_queries=result.get("rewritten_queries", [patient_text]),
        )

    def analyze_sync(self, patient_text: str, family_text: str = "", **kwargs) -> QueryAnalysis:
        import asyncio
        return asyncio.run(self.analyze(patient_text, family_text, **kwargs))
