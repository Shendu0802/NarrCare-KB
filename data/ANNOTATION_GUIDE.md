# NarrCare-KB Contraindication 标注指南

## 背景

本知识库用于安宁疗护场景的护理决策支持。当前 5441 个语义片段已通过 LLM 自动生成了摘要和多维标签，但**安全禁忌（contraindication）和风险等级（risk level）**仍需护理专业人员人工标注。

## 标注文件

`data/contraindication_annotation.csv` — 910 条候选片段（通过关键词和标签筛选的高风险相关内容）

## CSV 列说明

| 列名 | 说明 | 是否填写 |
|------|------|----------|
| `chunk_id` | 片段唯一 ID | 不要修改 |
| `source_book` | 来源书籍 | 参考 |
| `page` | 页码 | 参考 |
| `text_preview` | 文本前 300 字 | 阅读后判断 |
| `current_semantic_tags` | LLM 自动生成的标签 | 参考 |
| `current_risk_levels` | LLM 自动生成的风险等级 | 参考 |
| **`annotated_contraindications`** | ⚠️ **需填写** — 禁忌/注意事项 | 见下方 |
| **`annotated_risk_level`** | ⚠️ **需填写** — 正确风险等级 | 见下方 |
| `annotator_notes` | 标注备注（可选） | 自由文本 |

## 需要填写的字段

### 1. `annotated_contraindications` — 禁忌/注意事项

若片段包含以下任意内容，填写对应的禁忌标签（可多选，用逗号分隔）：

| 标签 | 含义 | 示例触发文本 |
|------|------|-------------|
| `吗啡禁忌` | 吗啡使用有呼吸抑制风险，需剂量控制 | "吗啡可能加速死亡"、"呼吸抑制" |
| `镇静禁忌` | 镇静/安眠药物在呼吸功能不全时慎用 | "安眠药加重呼吸困难" |
| `氧疗禁忌` | 过度吸氧可能导致 CO2 潴留 | "氧气越多越好"的误区 |
| `插管禁忌` | 临终患者不适用有创操作 | "家属坚持插管但患者拒绝" |
| `自杀风险` | 患者表达自杀意念或计划 | "不想活了"、"知道怎么结束" |
| `谵妄评估` | 谵妄需鉴别原因，不可简单镇静 | "胡言乱语"、"躁动不安" |
| `知情权冲突` | 家属隐瞒病情 vs 患者知情权 | "家属不让告诉患者" |
| `疼痛评估` | 疼痛需规范评估，不可凭经验给药 | "患者说疼但家属说忍忍" |
| `非药物优先` | 某些情况非药物干预优于药物 | "放松训练 vs 立即用药" |
| `文化敏感` | 涉及宗教/文化特殊需求 | "佛教徒不接受吗啡" |

若片段不涉及任何禁忌，**留空即可**（不要写"无"）。

### 2. `annotated_risk_level` — 风险等级

根据片段内容判断该知识用于临床决策时的风险：

| 等级 | 定义 | 示例 |
|------|------|------|
| `high` | 直接涉及生命安全的决策、药物使用禁忌、自杀风险 | 吗啡剂量、呼吸困难处理、自杀意念 |
| `medium` | 涉及护理质量、沟通策略、伦理冲突 | 病情告知方式、家属冲突处理 |
| `low` | 一般性知识、背景信息、理论描述 | 概念解释、研究背景 |

**注意**：如果 LLM 自动生成的 `current_risk_levels` 正确，直接复制即可；若不正确，填写你认为正确的等级。

## 标注示例

```
chunk_id: ku_pdfbook_abc123
text_preview: "临终呼吸困难可使用吗啡缓解，但需注意剂量，过量可能导致呼吸抑制..."
current_semantic_tags: 呼吸困难, 疼痛管理
current_risk_levels: high
→ annotated_contraindications: 吗啡禁忌
→ annotated_risk_level: high
→ annotator_notes: 确认风险等级为high，需强调剂量控制

chunk_id: ku_pdfbook_def456
text_preview: "护士可以通过叙事护理帮助患者表达未竟事务..."
current_semantic_tags: 叙事疗法, 未竟事务
current_risk_levels: medium
→ annotated_contraindications: （留空）
→ annotated_risk_level: low
→ annotator_notes: 该片段属于一般性护理建议，风险low
```

## 标注注意事项

1. **只填确实存在的禁忌** — 不要臆造标签。片段不涉及禁忌就留空。
2. **保守原则** — 不确定时宁可标 high，不可标 low。
3. **参考原文上下文** — text_preview 只有 300 字，如果不够判断可以看前后文逻辑。
4. **标注完成后保存为 CSV（UTF-8 编码）**，文件名为 `data/contraindication_annotated.csv`。

## 返回文件

标注完成后将文件放到：`data/contraindication_annotated.csv`

我会解析并回写到知识库，更新 safety recall 和 risk level 标签。
