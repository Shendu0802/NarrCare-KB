"""LLM enrichment: summary, tags, and knowledge card generation."""
from src.ingestion.tag_pool import validate_tags

ENRICH_SYSTEM_PROMPT = """You are a clinical knowledge curator for hospice and palliative care.
Given a text chunk from a medical/nursing document, generate:

1. summary: A concise Chinese summary (max 200 chars).
2. semantic_tags: Choose from the predefined tag list ONLY.
3. scenario_tags: Which care scenarios this applies to.
4. role_tags: Target roles (patient, family, nurse).
5. method_tags: Methods mentioned.
6. risk_levels: Risk level (low, medium, high).
7. card_targets: Card type(s) (mindfulness, healing, communication, personalized).

IMPORTANT: Only use tags that are provided in the system prompt. Do not invent new tags.

Predefined tags:
- semantic_tags: 死亡焦虑, 未竟事务, 预后不确定, 照护负担, 存在性痛苦, 生命意义, 告别与分离, 哀伤与丧失, 疼痛管理, 呼吸困难, 恶心呕吐, 疲劳与虚弱, 认知障碍, 谵妄, 抑郁情绪, 自杀意念, 灵性需求, 宗教需求, 文化敏感性, 家属沟通, 医疗决策, 预立医疗计划, 居家安宁, 住院安宁, 转诊决策, 儿童安宁, 老年安宁, 癌症末期
- scenario_tags: 夜间焦虑, 告别沟通, 治疗拒绝, 呼吸困难, 疼痛发作, 情绪崩溃, 家属冲突, 病情告知, 临终决策, 死亡准备, 哀伤辅导, 初次诊断, 复发告知, 转安宁病房, 出院计划
- role_tags: patient, family, nurse
- method_tags: 正念, 叙事疗法, 尊严疗法, 意义中心疗法, 认知行为疗法, 放松训练, 呼吸练习, 音乐疗法, 艺术疗法, 宠物疗法, 家庭会议, 动机访谈, 危机干预, 疼痛评估, 症状评估, 心理评估, 生命回顾, 遗愿清单, 告别仪式
- risk_levels: low, medium, high
- card_targets: mindfulness, healing, communication, personalized

Return valid JSON only."""

CARD_SYSTEM_PROMPT = """You are a clinical knowledge curator. Based on the provided text chunks,
create a knowledge card for nursing intervention. The card should be actionable for nurses.

Return JSON with:
- title: Card title
- text: Actionable nursing guidance (max 500 chars)
- card_targets: ["mindfulness"|"healing"|"communication"|"personalized"]

IMPORTANT: Do not fabricate medical advice. Only base content on the provided chunks."""


class LLMEnricher:
    """Generates summaries, tags, and knowledge cards using an LLM."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def enrich_chunk(self, chunk: dict) -> dict:
        messages = self.llm.build_messages(
            system=ENRICH_SYSTEM_PROMPT,
            user=f"Text:\n{chunk['text'][:2000]}",
        )
        try:
            result = await self.llm.chat_with_json(messages, temperature=0.3, max_tokens=1024)
        except Exception:
            result = {}

        tags = validate_tags(
            semantic_tags=result.get("semantic_tags", []),
            scenario_tags=result.get("scenario_tags", []),
            role_tags=result.get("role_tags", []),
            method_tags=result.get("method_tags", []),
            risk_levels=result.get("risk_levels", []),
            card_targets=result.get("card_targets", []),
        )
        return {**chunk, "summary": result.get("summary", ""), **tags}

    def enrich_chunk_sync(self, chunk: dict) -> dict:
        import asyncio
        return asyncio.run(self.enrich_chunk(chunk))

    async def generate_card(self, parent_chunks: list[dict]) -> dict | None:
        parent_ids = [c["id"] for c in parent_chunks]
        combined_text = "\n\n".join(c["text"][:500] for c in parent_chunks[:3])

        messages = self.llm.build_messages(system=CARD_SYSTEM_PROMPT, user=f"Source chunks:\n{combined_text}")
        try:
            result = await self.llm.chat_with_json(messages, temperature=0.5, max_tokens=2048)
        except Exception:
            return None

        if not result.get("title") or not result.get("text"):
            return None

        card_targets = [ct for ct in result.get("card_targets", [])
                        if ct in ("mindfulness", "healing", "communication", "personalized")]

        return {"title": result["title"], "text": result["text"], "card_targets": card_targets, "parent_chunk_ids": parent_ids}

    def generate_card_sync(self, parent_chunks: list[dict]) -> dict | None:
        import asyncio
        return asyncio.run(self.generate_card(parent_chunks))
