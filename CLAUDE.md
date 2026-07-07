# NarrCare-KB 项目手册

> 面向后续开发者/智能体的完整项目文档。阅读本文档即可理解项目全貌并接续开发。

---

## 1. 项目概述

**NarrCare-KB** 是 NarrCare 主系统的独立知识库服务，面向安宁疗护、死亡焦虑、叙事护理、家属沟通和护理干预场景。

### 核心目标
- 为 NarrCare 患者端、护士端、家属端输出和护士四卡片提供**可追溯的知识依据**
- 替代原有内置轻量 RAG 模块
- 独立部署、独立评测、独立优化

### 当前状态
| 维度 | 状态 |
|------|------|
| 整体进度 | Phase 1-7 全部实现 |
| 单元测试 | 28/28 passed |
| 检索评测 | 40 条真实场景，Recall@10 = 1.0 |
| 可部署 | 是 |

---

## 2. 技术栈

| 层级 | 选型 | 备注 |
|------|------|------|
| 后端框架 | Python 3.10 / FastAPI / Pydantic v2 / Uvicorn | venv 在 `NarrCare_KB/` |
| 向量检索 | FAISS (dense) + SQLite FTS5 (sparse) | IndexFlatIP，余弦相似度 |
| Embedding | 通义千问 text-embedding-v3（API） | 1024 维，batch_size=10 |
| Reranker | Qwen3-Reranker-0.6B（已下载，未启用） | GPU CC 不兼容 |
| LLM | 通义千问 qwen-plus / OpenAI-compatible | DashScope API |
| 文档解析 | PyMuPDF + easyocr | 扫描版 PDF OCR |
| 部署 | Linux GPU 服务器 (3×TITAN Xp, 62GB RAM) | Docker 未配置 |

### 关键版本约束
- **PyTorch 2.0.1+cu118**：TITAN Xp GPU 仅支持 CC 6.1，需 CUDA 11.8
- **transformers 4.36.2**：匹配 PyTorch 2.0
- **sentence-transformers 2.5.1**：匹配 transformers 4.36
- **numpy <2**：兼容旧版 torch
- **Qwen3-Embedding 本地模型已下载**但**未使用**（GPU 不兼容 + HF 不可达）
- **Qwen3-Reranker 本地模型已下载**但**未使用**（GPU 不兼容）

---

## 3. 项目结构

```
NarrCare_KB/
├── models/                              # 已下载的本地模型（未启用）
│   ├── Qwen3-Embedding-0.6B/
│   └── Qwen3-Reranker-0.6B/
├── 参考资料/                             # 10 本已导入的扫描 PDF 书籍
├── data/
│   ├── db/kb.sqlite                     # SQLite 数据库（~10MB, 5441 chunks）
│   ├── index/                           # FAISS 索引文件
│   │   ├── faiss.index                  # 向量索引 (~21MB)
│   │   ├── faiss_id_map.json            # FAISS ID → chunk ID 映射
│   │   ├── faiss_meta.json              # 索引元数据
│   │   └── faiss_vectorizer.pkl         # TF-IDF vectorizer（备用）
│   └── eval/
│       ├── rag_queries.jsonl            # 40 条评测用例
│       ├── ground_truth_candidates.json  # 评测 ground-truth 候选
│       └── report_*.md                  # 评测报告
├── src/
│   ├── main.py                          # FastAPI 入口，注册所有路由
│   ├── config.py                        # pydantic-settings 全局配置
│   ├── errors.py                        # KBException + KBErrorCode
│   ├── models/                          # 共享 Pydantic 模型
│   │   ├── knowledge_unit.py            # KnowledgeUnit, SourceType 等
│   │   ├── evidence_bundle.py           # EvidenceBundle, EvidenceItem
│   │   ├── retrieval.py                 # RetrieveRequest/Response
│   │   └── ingestion.py                 # IngestFilesRequest 等
│   ├── db/                              # 数据库层
│   │   ├── connection.py                # SQLite 连接管理 + schema
│   │   └── repository.py                # KnowledgeUnitRepository CRUD
│   ├── llm/                             # LLM 客户端
│   │   └── client.py                    # OpenAI-compatible async 客户端
│   ├── health/api.py                    # GET /health
│   ├── ingestion/                       # 入库模块
│   │   ├── api.py                       # POST /ingest/files, /ingest/search
│   │   ├── parser.py                    # PDF/MD/DOCX/JSONL 解析 + easyocr
│   │   ├── cleaner.py                   # 文本清洗 + 质量评分
│   │   ├── chunker.py                   # 语义切块
│   │   ├── enricher.py                  # LLM 标签/知识卡片生成（已实现，未运行）
│   │   ├── tag_pool.py                  # 30+ 预定义临床标签池
│   │   └── orchestrator.py              # 入库流水线编排
│   ├── retrieval/                       # 检索模块
│   │   ├── api.py                       # POST /retrieve
│   │   ├── query_understanding.py       # LLM query 分析 + 改写
│   │   ├── dense.py                     # FAISS 向量召回
│   │   ├── sparse.py                    # FTS5 全文召回（jieba 可选）
│   │   ├── metadata_recall.py           # 标签/角色元数据召回
│   │   ├── safety_recall.py             # 安全/禁忌规则召回
│   │   ├── hybrid.py                    # 多路融合 + 加权
│   │   ├── reranker.py                  # Qwen3 Reranker（已实现，未启用）
│   │   └── bundler.py                   # EvidenceBundle 组装
│   ├── indexing/                        # 索引模块
│   │   ├── api.py                       # POST /index/rebuild
│   │   ├── embedder.py                  # Embedder（API/TF-IDF 双模式）
│   │   ├── vector_index.py              # FAISS 构建/搜索/持久化
│   │   └── text_index.py                # FTS5 索引管理
│   ├── evaluation/                      # 评测模块
│   │   ├── api.py                       # POST /eval/run
│   │   ├── metrics.py                   # 7 个评测指标
│   │   └── report.py                    # 评测报告生成
│   └── debug/                           # 调试界面
│       ├── api.py                       # GET /debug/status, /debug/query
│       └── templates/debug.html         # 简易检索调试页
├── tests/
│   ├── test_config.py                   # 配置测试（3 tests）
│   ├── test_db.py                       # 数据库测试（5 tests）
│   ├── test_errors.py                   # 错误处理测试（3 tests）
│   ├── test_health.py                   # /health 端点测试（3 tests）
│   ├── test_llm_client.py               # LLM 客户端测试（4 tests）
│   ├── test_main.py                     # 应用入口测试（2 tests）
│   ├── test_models.py                   # 数据模型测试（8 tests）
│   ├── test_qwen_api.py                 # Qwen API 连通性测试（3 tests）
│   └── test_rebuild_index.py            # 索引重建脚本
├── scripts/
│   ├── import_local_files.py            # 并行批量导入 PDF（3 workers）
│   ├── discover_ground_truth.py         # 评测 ground-truth 发现
│   └── enrich_queries.py                # ground-truth ID 注入评测用例
├── docs/superpowers/
│   ├── specs/2026-07-02-narrcare-kb-design.md
│   └── plans/2026-07-02-narrcare-kb-plan.md
├── pyproject.toml
├── requirements.txt
├── .env                                 # 实际配置（含 API key，不提交）
├── .env.example                         # 配置模板
└── NarrCare_KB_Spec.md                  # 原始需求规格文档
```

---

## 4. 知识库数据概览

| 指标 | 数值 |
|------|------|
| 总 chunks | 5,441 |
| 来源类型 | 全部 `pdf_book` |
| 来源状态 | 全部 `main`（无 candidate/quarantined） |
| 单元类型 | 全部 `semantic_chunk`（无 knowledge_card） |
| 标签状态 | **全部为空**（LLM 增强未运行） |
| 文档数 | 10 本 |

### 已导入的 10 本书籍

| 书名 | 页数 | Chunks |
|------|------|--------|
| 善终守护师 | 217 | 176 |
| 存在主义心理治疗 | 583 | 1,087 |
| 学会告别：为临终做最好的安排 | 269 | 239 |
| 尊严疗法：临终寄语 | 220 | 382 |
| 毕淑敏自选集 生命卷 生命的栖息地 | 318 | 302 |
| 毕淑敏自选集 生命卷 西藏的故事 | 318 | 326 |
| 毕淑敏自选集 生命卷 预约死亡 | 318 | 312 |
| 生命的反转：急重症科医生手记 | 236 | 391 |
| 生命的礼物 关于爱死亡及存在的意义 | 285 | 975 |
| 直视骄阳：征服死亡恐惧 | 291 | 994 |

> 所有 PDF 均为扫描版（无文本层），通过 easyocr CPU 模式识别（DPI 120，~4.7s/页）。

---

## 5. API 清单

| 方法 | 路径 | 状态 | 说明 |
|------|------|------|------|
| GET | `/health` | ✅ | 健康检查 |
| POST | `/retrieve` | ✅ | 知识检索（返回 EvidenceBundle） |
| POST | `/ingest/files` | ✅ | 手动导入本地文件 |
| POST | `/ingest/search` | ⚠️ | 自动扩充（返回 not_implemented） |
| POST | `/index/rebuild` | ✅ | 重建 FAISS + FTS5 索引 |
| POST | `/eval/run` | ✅ | 运行评测并生成报告 |
| GET | `/debug/status` | ✅ | 调试状态 |
| GET | `/debug/query` | ✅ | 调试检索页面 |

### /retrieve 请求体
```json
{
  "session_id": "",
  "patient_text": "患者晚上睡不着害怕死亡",
  "family_text": "",
  "user_role": "nurse",
  "risk_assessment": {},
  "dyadic_analysis": {},
  "top_k_cards": 3,
  "top_k_passages": 5,
  "include_debug": false
}
```

### /retrieve 响应体
返回 `EvidenceBundle`：包含 `query_analysis`、`card_sources`（mindfulness/healing/communication/personalized）、`supporting_passages`、`safety_and_boundary_evidence`、`candidate_evidence`、`excluded_evidence`。

---

## 6. 启动方式

```bash
cd /home/junyi/Documents/NarrCare_KB
source NarrCare_KB/bin/activate

# 运行测试
python -m pytest tests/ -v --ignore=tests/test_qwen_api.py --ignore=tests/test_rebuild_index.py

# 启动服务（默认 9000 端口）
uvicorn src.main:app --host 0.0.0.0 --port 9000

# 批量导入 PDF
python scripts/import_local_files.py

# 重建 FAISS 索引（使用 Qwen API embedding）
python tests/test_rebuild_index.py

# 发现评测 ground-truth
python scripts/discover_ground_truth.py

# 运行评测
python -c "import asyncio; from src.evaluation.api import run_eval; asyncio.run(run_eval())"
```

---

## 7. 配置说明

配置由 `pydantic-settings` 管理，环境变量前缀 `KB_`。优先级：环境变量 > `.env` 文件 > 代码默认值。

### .env 关键配置（已设置为通义千问）

```bash
KB_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
KB_LLM_API_KEY=sk-5a858dfd8e3c4e25bdd54d7431346e19
KB_LLM_MODEL=qwen-plus
KB_LLM_TIMEOUT=60
KB_EMBEDDING_DEVICE=cpu          # GPU 不可用，设为 cpu
KB_FAISS_INDEX_PATH=data/index/faiss.index
KB_DB_PATH=data/db/kb.sqlite
```

### src/llm/client.py 硬编码默认值
`LLMConfig` 类有硬编码的 API key 和 model 默认值。如果 `.env` 中 `KB_LLM_API_KEY` 为空，会使用 `LLMConfig` 的默认值。**注意：当通过 `settings.llm_api_key` 空字符串构造 LLMConfig 时，空字符串会覆盖默认值导致鉴权失败。** `retrieval/api.py` 已处理此情况。

---

## 8. 评测系统

### 指标（7 个）
| 指标 | 当前值 | 目标 | 状态 |
|------|--------|------|------|
| Recall@10 | 1.0 | >= 0.70 | ✅ |
| Safety Hit Rate | 0.15 | >= 0.95 | ❌ |
| Noise Rate | 0.0 | <= 0.15 | ✅ |
| Tag Match | 0.0 | N/A | — |
| Source Type Match | 1.0 | >= 0.80 | ✅ |
| Flag Check | 1.0 | >= 0.95 | ✅ |
| Card Target Match | 0.075 | N/A | — |

### 评测流程
1. `scripts/discover_ground_truth.py` — 对 40 条 query 运行 dense+sparse 召回，输出候选 chunks
2. 自动标记 top-3 dense hits 为 relevant，填入 `must_include_ids`
3. `POST /eval/run` — 逐条检索，计算 7 个指标，生成 markdown 报告
4. 报告保存在 `data/eval/report_*.md`

### 注意
- 评测使用 `_retrieve_direct()` 函数，跳过 LLM query rewriting，确保确定性
- Recall=1.0 是因为 ground-truth 由同一检索管道发现（正向自检），需人工验证后才有独立意义
- Safety=0.15 是因为所有 chunk 的 contraindications 标签为空

---

## 9. 已知问题与局限

### 严重
1. **GPU 不可用**：TITAN Xp (CC 6.1) 与 PyTorch 2.12.1 (需 CC 7.5+) 不兼容。已安装 PyTorch 2.0.1+cu118 但 sentence-transformers 版本冲突，最终放弃本地 embedding/reranker
2. **所有 chunk 标签为空**：LLM 入库增强 (`enricher.py`) 未运行，导致 semantic_tags/scenario_tags/role_tags/card_targets 均为空
3. **无 knowledge_card**：所有单元为 semantic_chunk，无 LLM 生成的知识卡片

### 中等
4. **Qwen3-Embedding 本地模型闲置**：已下载至 `models/` 但无法加载（HuggingFace 不可达 + transformers 版本冲突）
5. **Qwen3-Reranker 本地模型闲置**：同上，`reranker.py` 已实现但未集成到检索管道
6. **仅 pdf_book 来源**：无 guideline/paper/case 等多源内容
7. **OCR 质量**：easyocr CPU 模式识别扫描 PDF，部分文本含乱码

### 轻微
8. **jieba 未安装**：sparse recall 退化为不分词模式，中文检索精度降低
9. **/ingest/search 未实现**：自动扩充功能仅返回 not_implemented
10. **Docker 未配置**：仅有 requirements.txt，无 Dockerfile

---

## 10. 待办事项（优先级排序）

### P0 — 必须完成
- [ ] **安装兼容 PyTorch** 以启用 GPU：`torch>=2.4` 配合 `transformers>=4.43`（需 CUDA 12.x 或更换 GPU）
  - 或：使用通义千问 API 替代本地 embedding（已实现 `Embedder("api")` 模式）
- [ ] **运行 LLM 入库增强**：调用 `enricher.py` 为 5441 chunks 生成摘要、标签和知识卡片

### P1 — 应该完成
- [ ] 集成 Reranker：将 `reranker.py` 加入检索管道
- [ ] 安装 jieba：`pip install jieba` 改善中文 FTS5 召回
- [ ] 扩展知识来源：导入 guideline、paper、case 等多源内容
- [ ] 添加 quarantined 样本：测试 noise_rate 和 flag_check 指标

### P2 — 可以完成
- [ ] 实现 /ingest/search 自动扩充
- [ ] 人工验证评测 ground-truth（当前为自动标记）
- [ ] Docker 化部署
- [ ] 增量索引更新（当前仅支持全量重建）

---

## 11. Git 提交历史

```
aa5aab4 feat: comprehensive eval — 40 realistic patient queries, 7 metrics, report gen
e5826a3 feat: Qwen DashScope semantic embeddings replace TF-IDF
7a808f1 feat: TF-IDF embedder, API retrieval pipeline working end-to-end
c48d92b fix: parallel import with imap_unordered, SQLite busy timeout
f88632a fix: integrate easyocr for scanned PDFs, lazy-load jinja2 templates
56325d2 feat: complete NarrCare-KB implementation — all 7 phases
e5c67cf feat: add database layer with SQLite schema and repository
3549a25 feat: add Pydantic data models (KnowledgeUnit, EvidenceBundle, requests)
e295b49 feat: add unified error handling with KBException
b8ad8af feat: add configuration management with pydantic-settings
a4d1ad1 feat: project scaffolding with dependencies and env template
```

---

## 12. 快速接续指南

### 如果你想：
- **启动服务测试检索** → 第 6 节
- **了解 API 接口** → 第 5 节
- **运行评测看效果** → 第 8 节
- **了解技术债务** → 第 9-10 节
- **理解设计决策** → `docs/superpowers/specs/2026-07-02-narrcare-kb-design.md`
- **了解原始需求** → `NarrCare_KB_Spec.md`
- **查看详细实施计划** → `docs/superpowers/plans/2026-07-02-narrcare-kb-plan.md`
