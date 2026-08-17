# NarrCare-KB

独立知识库服务，面向**安宁疗护、死亡焦虑、叙事护理、家属沟通和护理干预**场景，为 NarrCare 主系统（患者端、护士端、家属端 + 护士四卡片）提供高质量、可解释、可追溯的知识检索与证据组织能力。

## 特性

- 🔍 **混合检索** — Dense（FAISS 向量）+ Sparse（FTS5 全文）+ Metadata（标签）+ Safety（禁忌规则）四路召回
- 🧠 **Query Understanding** — LLM 分析意图、场景、风险信号并改写查询
- 🎯 **Reranker** — DeepSeek API 精排，无需 GPU
- 🗂️ **分层证据包** — `card_sources`（四卡片）+ `supporting_passages` + `safety_and_boundary_evidence`
- 📚 **可追溯** — 每个知识片段保留来源、页码、质量分；知识卡片关联 `parent_chunk_ids`
- 🏷️ **多维标签** — semantic / scenario / role / method / risk / card_target 六维标签
- 📊 **评测系统** — 40 条真实病患场景，7 项指标

## 数据概览

| 指标 | 数值 |
|------|------|
| 总单元数 | 15,073 |
| 语义片段 | 11,669（5,441 书籍 + 6,012 指南 + 216 论文） |
| 知识卡片 | 3,404 |
| 文档数 | 46（10 本书 + 36 份指南/论文） |
| LLM 标签覆盖 | 71% |

**知识来源：** 10 本安宁疗护/存在主义心理学著作 + 36 份临床指南（国家卫健委 2025 指南、CACA 癌痛/心理指南、NICE NG31、WHO 癌痛指南、NCCN/ASCO 等）

## 技术栈

| 层级 | 选型 |
|------|------|
| 后端 | Python 3.10 / FastAPI / Pydantic v2 / Uvicorn |
| 向量检索 | FAISS (IndexFlatIP) + SQLite FTS5 |
| Embedding | Qwen text-embedding-v3 API（备用 TF-IDF） |
| Reranker | DeepSeek API 精排 |
| LLM | DeepSeek / 通义千问（OpenAI-compatible） |
| OCR | PyMuPDF + easyocr（扫描 PDF） |
| 存储 | SQLite（~41MB） |

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv NarrCare_KB
source NarrCare_KB/bin/activate
pip install -r requirements.txt

# 2. 配置 API（复制 .env.example 为 .env 并填入 key）
cp .env.example .env

# 3. 运行测试
python -m pytest tests/ -v --ignore=tests/test_qwen_api.py --ignore=tests/test_rebuild_index.py

# 4. 启动服务
uvicorn src.main:app --host 0.0.0.0 --port 9000
```

## API 使用

### 健康检查
```bash
curl http://localhost:9000/health
```

### 知识检索
```bash
curl -X POST http://localhost:9000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "患者晚上睡不着，说害怕闭上眼睛就再也醒不来了",
    "family_text": "女儿每晚陪护也很焦虑",
    "user_role": "nurse",
    "top_k_cards": 3,
    "top_k_passages": 5
  }'
```

响应返回 `EvidenceBundle`，包含：

| 字段 | 用途 |
|------|------|
| `query_analysis` | LLM 查询理解（意图/场景/风险/改写） |
| `card_sources` | 四卡片来源（mindfulness/healing/communication/personalized） |
| `supporting_passages` | 知识依据片段 |
| `safety_and_boundary_evidence` | 安全/禁忌证据（必须进入 prompt） |
| `candidate_evidence` | 候选证据（低确定性表达） |
| `excluded_evidence` | 排除证据（仅 debug） |

### 其他端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ingest/files` | 手动导入本地文件 |
| POST | `/index/rebuild` | 重建 FAISS + FTS5 索引 |
| POST | `/eval/run` | 运行评测生成报告 |
| GET | `/debug/query` | 调试检索页面 |
| GET | `/debug/status` | 调试状态 |

## 数据流水线

```
入库：PDF/MD/DOCX/HTML → 解析(OCR) → 清洗 → 质量评分 → 语义切块
      → LLM 标签/摘要/知识卡片 → 向量化 → FAISS/FTS5 索引

检索：用户文本 → LLM Query Understanding → 多路召回(Dense/Sparse/Metadata/Safety)
      → Hybrid 融合 → Reranker 精排 → EvidenceBundle 输出
```

## 评测

40 条真实病患/家属/护士场景，覆盖死亡焦虑、夜间恐惧、家属否认、照护负担、未竟事务、告别沟通、治疗拒绝、预后不确定、呼吸困难、高风险绝望。

| 指标 | 当前值 | 目标 |
|------|--------|------|
| Recall@10 | 1.0 | ≥ 0.70 |
| Tag Match | 0.79 | — |
| Card Target Match | 0.96 | — |
| Source Type Match | 1.0 | ≥ 0.80 |
| Flag Check | 1.0 | ≥ 0.95 |
| Safety Hit Rate | 0.15 | ≥ 0.95（待人工标注） |

## 项目结构

```
src/
├── main.py               # FastAPI 入口
├── config.py             # 配置（pydantic-settings）
├── ingestion/            # 入库：解析/清洗/切块/LLM增强/编排
├── retrieval/            # 检索：query理解/多路召回/rerank/证据包
├── indexing/             # 索引：embedding/FAISS/FTS5
├── evaluation/           # 评测：7 指标/报告
├── debug/                # 调试界面
└── llm/                  # LLM 客户端（OpenAI-compatible）
scripts/                  # 批量导入/ground-truth/卡片生成脚本
tests/                    # 单元测试（28 passed）
```

## 已知限制

- GPU 不可用（TITAN Xp CC 6.1 与 PyTorch 不兼容），embedding/reranker 走 API
- guideline 的 LLM 增强进行中（32% 完成）
- Safety Hit Rate 0.15 — 等待护理同学人工标注 contraindications
- Qwen embedding API 曾因欠费临时改用 TF-IDF，恢复需 `python tests/test_rebuild_index.py`

## 文档

- 项目手册（详细）：`CLAUDE.md`
- 原始需求规格：`NarrCare_KB_Spec.md`
- 架构设计：`docs/superpowers/specs/2026-07-02-narrcare-kb-design.md`
- 实施计划：`docs/superpowers/plans/2026-07-02-narrcare-kb-plan.md`

## License

© NarrCare 项目组。仅供内部研究与开发使用。
