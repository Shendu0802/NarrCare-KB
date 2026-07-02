"""Predefined tag pool for LLM enrichment validation."""

TAG_POOL = {
    "semantic_tags": [
        "死亡焦虑", "未竟事务", "预后不确定", "照护负担",
        "存在性痛苦", "生命意义", "告别与分离", "哀伤与丧失",
        "疼痛管理", "呼吸困难", "恶心呕吐", "疲劳与虚弱",
        "认知障碍", "谵妄", "抑郁情绪", "自杀意念",
        "灵性需求", "宗教需求", "文化敏感性",
        "家属沟通", "医疗决策", "预立医疗计划",
        "居家安宁", "住院安宁", "转诊决策",
        "儿童安宁", "老年安宁", "癌症末期",
    ],
    "scenario_tags": [
        "夜间焦虑", "告别沟通", "治疗拒绝", "呼吸困难",
        "疼痛发作", "情绪崩溃", "家属冲突", "病情告知",
        "临终决策", "死亡准备", "哀伤辅导", "初次诊断",
        "复发告知", "转安宁病房", "出院计划",
    ],
    "role_tags": ["patient", "family", "nurse"],
    "method_tags": [
        "正念", "叙事疗法", "尊严疗法", "意义中心疗法",
        "认知行为疗法", "放松训练", "呼吸练习",
        "音乐疗法", "艺术疗法", "宠物疗法",
        "家庭会议", "动机访谈", "危机干预",
        "疼痛评估", "症状评估", "心理评估",
        "生命回顾", "遗愿清单", "告别仪式",
    ],
    "risk_levels": ["low", "medium", "high"],
    "card_targets": ["mindfulness", "healing", "communication", "personalized"],
}


def validate_tags(**tag_fields: list[str]) -> dict[str, list[str]]:
    """Filter tags to only include those in the predefined pool."""
    valid = {}
    for field_name, tags in tag_fields.items():
        pool = TAG_POOL.get(field_name, [])
        if pool:
            valid[field_name] = [t for t in tags if t in pool]
        else:
            valid[field_name] = tags
    return valid
