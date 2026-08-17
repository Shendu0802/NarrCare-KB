# NarrCare-KB 独立知识库服务项目 Spec

## 1. 项目目标

`NarrCare-KB` 是 NarrCare 项目的独立知识库服务，用于替代当前主系统内置的轻量 RAG 模块。它面向安宁疗护、死亡焦虑、叙事护理、家属沟通和护理干预场景，提供高质量、可解释、可追溯的知识检索与证据组织能力。

该服务的核心目标不是做通用问答，而是为 NarrCare 三端输出和护士端四卡片提供可靠知识依据。

最终效果应解决当前模块的主要问题：

- 用户输入为模糊自然语言，直接关键词/BM25 检索效果不稳定。
- 现有知识库内容贫瘠，需要支持 PDF、论文、指南、案例和公开权威资料自动扩充。
- PDF OCR 和文本抽取质量不稳定，需要更强清洗、质量评分和隔离机制。
- 当前 RAG 返回 top-k 片段，缺少来源分类、语义标签、卡片映射和证据层级。
- 当前模块与主系统耦合，需要独立部署、独立评测、独立优化。

## 2. 总体架构

采用独立服务架构：

```text
NarrCare 主系统
  -> 调用 NarrCare-KB /retrieve
  -> 获得 EvidenceBundle
  -> 生成患者端、护士端、家属端输出和护士四卡片

NarrCare-KB 独立服务
  -> 资料导入/自动扩充
  -> 文档解析与清洗
  -> 语义切块
  -> LLM 摘要/标签/知识卡片生成
  -> GPU embedding
  -> FAISS 向量索引
  -> SQLite 元数据与全文索引
  -> hybrid recall
  -> reranker 重排
  -> 分层证据包输出
```

部署目标：

- Linux GPU 服务器。
- 单卡 8-12GB 显存可运行。
- GPU 主要用于 embedding、reranker 和批量索引构建。
- LLM 相关能力调用外部 OpenAI-compatible API，不本地部署大语言模型。

第一阶段不保留旧 RAG fallback。若 `NarrCare-KB` 服务不可用、索引为空、模型加载失败或返回 schema 非法，NarrCare 主系统应显式报错，方便排查。

## 3. 技术选型

后端服务：

- Python 3.11+
- FastAPI
- Pydantic v2
- Uvicorn

向量与检索：

- FAISS 作为 dense vector index。
- SQLite 作为元数据、来源、标签、知识卡片和评测记录存储。
- SQLite FTS5 或独立 BM25 实现作为 sparse recall。
- Hybrid recall = dense recall + sparse recall + metadata/rule recall。
- Reranker 使用 Qwen3 reranker。

模型：

- 默认 embedding：`Qwen3-Embedding-0.6B`
- 默认 reranker：`Qwen3-Reranker-0.6B`
- 备选 embedding：`BAAI/bge-m3`
- 备选 reranker：`BAAI/bge-reranker-v2-m3`
- LLM：外部 OpenAI-compatible API，例如 DeepSeek、通义千问、OpenAI-compatible 私有服务等。

文档解析：

- PyMuPDF 用于 PDF 文本层抽取。
- PaddleOCR 或可替换 OCR 后端用于扫描页 fallback。
- 不做复杂版面工程，但必须加强清洗、去噪、质量评分、目录/版权页过滤和低质量隔离。

调试界面：

- 可使用 Streamlit 或 FastAPI 简易 HTML 页面。
- 只做研发调试，不做完整管理后台。

## 4. 知识来源与分类体系

### 4.1 支持的资料来源

知识库应支持以下资料类型：

- 本地 PDF 书籍；
- 论文 PDF；
- 临床指南或护理指南；
- Markdown 文档；
- Docx 文档；
- 现有结构化四库 JSONL；
- 病例/模拟案例 JSONL；
- 自动检索得到的公开权威资料。

自动扩充优先来源：

- PubMed / PMC；
- WHO 等公开权威组织；
- NCCN 或同级别公开指南源；
- 安宁疗护、叙事医学、死亡焦虑、照护者支持相关公开资料。

不处理需要登录、付费、版权不明确或抓取限制明显的资料。

### 4.2 分类原则

一级分类按资料来源管理，便于维护和引用：

```text
pdf_book
paper
guideline
case
structured_library
markdown
web_reference
```

每个知识单元同时具备多维标签：

```text
semantic_tags      # 死亡焦虑、未竟事务、预后不确定、照护负担等
scenario_tags      # 夜间焦虑、告别沟通、治疗拒绝、呼吸困难等
role_tags          # patient, family, nurse
method_tags        # 正念、叙事疗法、尊严疗法、意义中心疗法、沟通策略等
risk_levels        # low, medium, high
card_targets       # mindfulness, healing, communication, personalized
```

这样既保留来源管理清晰度，也支持语义检索、场景过滤和四卡片映射。

## 5. 知识单元设计

知识库采用双层知识单元：

1. `semantic_chunk`
   - 来自原始资料的可引用语义片段。
   - 是检索、引用、溯源和证据支撑的基础。
   - 必须保留来源、页码、原文、质量分和解析方法。

2. `knowledge_card`
   - 由 LLM 基于一个或多个 `semantic_chunk` 生成。
   - 面向护理干预、沟通建议和四卡片生成。
   - 必须保留 `parent_chunk_ids`，不能脱离原文证据链。

核心 schema：

```json
{
  "id": "ku_xxx",
  "unit_type": "semantic_chunk | knowledge_card",
  "source_type": "pdf_book | paper | guideline | case | structured_library | markdown | web_reference",
  "source_status": "main | candidate | quarantined",
  "title": "",
  "text": "",
  "summary": "",
  "source_uri": "",
  "source_title": "",
  "source_citation": "",
  "page_start": null,
  "page_end": null,
  "semantic_tags": [],
  "scenario_tags": [],
  "role_tags": ["patient", "family", "nurse"],
  "method_tags": [],
  "risk_levels": [],
  "card_targets": ["mindfulness", "healing", "communication", "personalized"],
  "contraindications": [],
  "quality_score": 0.0,
  "review_status": "unreviewed | approved | rejected",
  "parent_chunk_ids": [],
  "embedding_model": "",
  "created_at": "",
  "updated_at": ""
}
```

第一阶段策略：

- 手动导入资料默认 `source_status=main`。
- 自动扩充资料默认 `source_status=candidate`。
- 低质量、乱码、目录页、无信息片段进入 `source_status=quarantined`。
- `candidate` 可参与检索，但权重低于 `main`，并必须在返回结果中明确标记。
- 暂不做人工审核后台，但保留 `review_status`、`quality_score`、`notes` 等字段，方便后续升级。

## 6. 入库流程

入库 pipeline：

```text
source acquisition
  -> document parsing
  -> text cleanup
  -> quality scoring
  -> semantic chunking
  -> metadata tagging
  -> LLM summary/card generation
  -> embedding
  -> FAISS/SQLite index update
```

### 6.1 资料获取

支持两种模式：

- 手动导入：用户上传或放置本地文件。
- 自动扩充：根据主题词、检索词、领域标签，从公开权威源检索资料。

自动扩充流程：

1. 用户提供主题，例如“death anxiety hospice caregiver communication”。
2. 系统检索公开权威资料。
3. 保存来源 URL、标题、摘要、获取时间。
4. 下载可访问全文或摘要。
5. 入库为 `candidate`。
6. 标记 `source_type=paper | guideline | web_reference`。

### 6.2 文档解析与清洗

必须处理：

- PDF 文本层抽取；
- 扫描页 OCR fallback；
- 页眉页脚去除；
- 目录页、版权页、索引页过滤；
- 孤立字符、乱码、低信息密度文本过滤；
- 重复片段去重；
- 质量评分。

质量评分字段：

```text
quality_score: 0.0 - 1.0
quality_flags:
  - too_short
  - table_of_contents
  - copyright_page
  - isolated_characters
  - possible_ocr_garble
  - duplicated
  - low_information_density
```

进入最终主检索的最低要求：

- `quality_score >= 0.5`
- 不包含严重乱码或孤立字符标记
- 不属于目录页、版权页、索引页

低于要求的片段进入 `quarantined`。

### 6.3 语义切块

切块应优先按语义边界，而不是固定字符数硬切。

默认策略：

- 先按标题、段落、页码、自然段切分。
- 再合并过短段落。
- 每个 chunk 建议 300-800 中文字。
- overlap 控制在 80-150 中文字。
- 保留页码和原始位置。

### 6.4 LLM 入库增强

对每个高质量 `semantic_chunk`，调用外部 LLM 生成：

- `summary`
- `semantic_tags`
- `scenario_tags`
- `role_tags`
- `method_tags`
- `risk_levels`
- `card_targets`
- 可选 `knowledge_card`

LLM 生成内容必须满足：

- 不得删除原文引用。
- 不得产生无来源护理结论。
- 每张 `knowledge_card` 必须指向一个或多个 `parent_chunk_ids`。
- LLM 输出失败时，chunk 仍可入库，只是缺少自动标签或卡片。

## 7. 检索流程

`POST /retrieve` 不直接对用户原文做单路向量检索，而是分阶段处理。

### 7.1 Query Understanding

输入：

- 患者文本；
- 家属文本；
- 用户角色；
- 风险评分结果；
- 会话上下文；
- 可选已有 dyadic analysis。

外部 LLM 生成：

```json
{
  "intents": [],
  "scenario_tags": [],
  "role_focus": [],
  "risk_signals": [],
  "contraindication_signals": [],
  "rewritten_queries": []
}
```

`rewritten_queries` 至少包括：

- 原始表达保留版；
- 场景化查询；
- 护理干预查询；
- 安全/禁忌查询。

### 7.2 Hybrid Recall

召回来源：

- dense vector recall：FAISS；
- sparse recall：BM25/FTS；
- metadata recall：场景、角色、风险等级、card_targets；
- safety recall：高风险和禁忌相关规则强制召回。

召回时的权重原则：

- `main` > `candidate` > `quarantined`
- `quarantined` 默认不进入最终证据包，只能进入 debug/excluded。
- 与 `card_targets` 匹配的知识卡片优先用于四卡片来源。
- 指南、论文、结构化库在安全/边界证据中优先级高于普通书籍片段和案例。

### 7.3 Rerank

对召回候选使用 `Qwen3-Reranker-0.6B` 重排。

重排输入应包含：

- 用户原始文本；
- query analysis；
- 候选知识文本；
- 候选标签；
- 来源状态；
- 质量分。

最终排序综合：

```text
final_score =
  rerank_score
  + source_status_weight
  + quality_score_weight
  + metadata_match_weight
  + safety_priority_boost
```

具体权重可以在配置中调整，默认：

```text
rerank_score: 0.55
dense/sparse recall score: 0.20
metadata match: 0.10
quality score: 0.10
source status: 0.05
```

安全/禁忌规则不完全依赖分数，必要时强制进入 `safety_and_boundary_evidence`。

## 8. 输出接口

`POST /retrieve` 返回分层证据包，而不是旧 top-k list。

```json
{
  "query_analysis": {
    "intents": [],
    "scenario_tags": [],
    "role_focus": [],
    "risk_signals": [],
    "contraindication_signals": [],
    "rewritten_queries": []
  },
  "card_sources": {
    "mindfulness": [],
    "healing": [],
    "communication": [],
    "personalized": []
  },
  "supporting_passages": [],
  "safety_and_boundary_evidence": [],
  "candidate_evidence": [],
  "excluded_evidence": [],
  "retrieval_debug": {
    "dense_hits": [],
    "sparse_hits": [],
    "metadata_hits": [],
    "reranked_hits": [],
    "model_versions": {},
    "index_version": ""
  }
}
```

每条 evidence item 使用统一结构：

```json
{
  "id": "",
  "unit_type": "semantic_chunk | knowledge_card",
  "source_type": "",
  "source_status": "main | candidate | quarantined",
  "title": "",
  "snippet": "",
  "summary": "",
  "score": 0.0,
  "source_citation": "",
  "source_uri": "",
  "page_start": null,
  "page_end": null,
  "semantic_tags": [],
  "scenario_tags": [],
  "role_tags": [],
  "method_tags": [],
  "risk_levels": [],
  "card_targets": [],
  "quality_score": 0.0,
  "review_status": "",
  "parent_chunk_ids": []
}
```

NarrCare 主系统消费规则：

- `card_sources.mindfulness` 用于正念卡片；
- `card_sources.healing` 用于疗愈卡片；
- `card_sources.communication` 用于沟通卡片；
- `card_sources.personalized` 用于个性化建议和安全提示；
- `supporting_passages` 用于知识依据展示；
- `safety_and_boundary_evidence` 必须进入 prompt；
- `candidate_evidence` 可作为补充，但生成时需降低确定性表达；
- `excluded_evidence` 只用于 debug，不进入用户输出。

## 9. API 清单

必须实现：

```text
GET  /health
POST /retrieve
POST /ingest/files
POST /ingest/search
POST /index/rebuild
POST /eval/run
GET  /debug/status
GET  /debug/query
```

### 9.1 `GET /health`

返回：

```json
{
  "status": "ok | degraded | error",
  "gpu_available": true,
  "embedding_model": "",
  "reranker_model": "",
  "index_loaded": true,
  "index_version": "",
  "document_count": 0,
  "chunk_count": 0,
  "card_count": 0
}
```

### 9.2 `POST /retrieve`

输入：

```json
{
  "session_id": "",
  "patient_text": "",
  "family_text": "",
  "user_role": "patient | family | nurse",
  "risk_assessment": {},
  "dyadic_analysis": {},
  "top_k_cards": 3,
  "top_k_passages": 5,
  "include_debug": false
}
```

输出：`EvidenceBundle`。

### 9.3 `POST /ingest/files`

输入本地文件路径或上传文件，返回入库任务状态。

### 9.4 `POST /ingest/search`

输入主题和来源约束：

```json
{
  "topic": "",
  "keywords": [],
  "source_priority": ["pubmed", "pmc", "who", "guideline"],
  "max_results": 20
}
```

返回候选资料列表和入库任务状态。

### 9.5 `POST /index/rebuild`

重建 FAISS 和 sparse index。支持全量重建，后续可扩展增量重建。

### 9.6 `POST /eval/run`

运行评测集并返回指标。

## 10. 调试界面

调试界面不做完整后台，只服务研发、答辩和效果调参。

必须包含：

- 检索输入框；
- query analysis 展示；
- dense/sparse/metadata/rerank 各阶段结果；
- 最终 EvidenceBundle；
- 来源类型统计；
- main/candidate/quarantined 数量；
- 索引状态；
- 模型版本；
- 评测指标展示。

不要求第一阶段支持：

- 人工审核流；
- 批量标签编辑；
- 多用户权限；
- 在线删除/修改资料。

## 11. 与 NarrCare 主系统集成

主系统新增配置：

```text
KB_SERVICE_URL=http://127.0.0.1:9000
KB_TIMEOUT_SECONDS=30
```

集成规则：

- `/analyze` 调用 `KB_SERVICE_URL/retrieve`。
- 成功时将 `EvidenceBundle` 放入 prompt。
- 失败时返回明确错误，例如：
  - `KB_SERVICE_UNAVAILABLE`
  - `KB_INDEX_NOT_READY`
  - `KB_SCHEMA_INVALID`
  - `KB_RETRIEVAL_TIMEOUT`
- 不自动回退旧 RAG。
- 旧 `src/rag` 可以保留为历史模块，但不再用于主流程。

## 12. 评测与验收标准

建立 `data/eval/rag_queries.jsonl`，覆盖至少以下场景：

- 死亡焦虑；
- 夜间恐惧；
- 家属否认或 minimization；
- 照护者负担；
- 未竟事务；
- 告别沟通；
- 治疗拒绝；
- 预后不确定；
- 呼吸困难禁忌；
- 高风险绝望表达。

每条评测样例包含：

```json
{
  "query": "",
  "patient_context": "",
  "family_context": "",
  "expected_tags": [],
  "must_include_source_types": [],
  "must_include_card_targets": [],
  "must_include_ids": [],
  "must_not_include_flags": [],
  "risk_level": "",
  "human_notes": ""
}
```

宽松起步验收指标：

- `Recall@10 >= 0.70`
- 安全/禁忌规则命中率 `>= 95%`
- OCR/目录/乱码噪声进入最终证据包比例 `<= 15%`
- 人工可用性评分均分 `>= 3.5 / 5`
- 每个主要 demo case 至少返回：
  - 1 条四卡片主来源；
  - 1 条支撑片段；
  - 1 条安全或边界证据。
- `/retrieve` 在索引已加载情况下必须返回合法 `EvidenceBundle`。
- 模型、索引、外部 LLM 或检索异常必须返回可诊断错误。

## 13. 实施阶段建议

### Phase 1：服务骨架与数据模型

- 建立独立 FastAPI 服务。
- 建立 SQLite schema。
- 实现基础 `KnowledgeUnit`、`EvidenceBundle` schema。
- 实现 `/health`、`/retrieve` mock、`/debug/status`。
- 主系统接入 KB 服务并移除旧 RAG fallback。

### Phase 2：文档入库与清洗

- 实现文件导入。
- 实现 PDF/Markdown/JSONL/case 解析。
- 实现质量评分、目录页过滤、低质量隔离。
- 写入 SQLite。

### Phase 3：LLM 标签与知识卡片

- 接入外部 OpenAI-compatible API。
- 为 chunk 生成摘要、标签和知识卡片。
- 确保知识卡片保留 `parent_chunk_ids`。

### Phase 4：GPU 向量化与索引

- 接入 Qwen3 embedding。
- 批量生成向量。
- 建立 FAISS index。
- 建立 sparse recall。
- 实现 `/index/rebuild`。

### Phase 5：Query Understanding 与 Hybrid Retrieval

- 接入外部 LLM 做 query understanding。
- 实现 query rewrite。
- 实现 dense/sparse/metadata/safety 多路召回。
- 实现 source_status、quality_score、metadata 权重。

### Phase 6：Reranker 与分层证据包

- 接入 Qwen3 reranker。
- 组装 `card_sources`、`supporting_passages`、`safety_and_boundary_evidence`、`candidate_evidence`、`excluded_evidence`。
- 输出 debug 信息。

### Phase 7：自动扩充与评测

- 实现权威公开源自动检索。
- 自动扩充资料进入 candidate 层。
- 建立评测集。
- 实现 `/eval/run` 和调试界面指标展示。

## 14. 非目标

第一阶段不做：

- 完整人工审核后台；
- 多用户权限系统；
- 大规模分布式向量数据库；
- 本地大语言模型部署；
- 复杂 PDF 版面理解系统；
- 医学诊断或治疗建议生成；
- 替代医生、护士或心理治疗师判断。

## 15. 风险与约束

主要风险：

- 外部 LLM API 不稳定会影响 query understanding 和入库增强。
- 自动扩充资料可能相关性不够，需要 candidate 层和低权重策略控制风险。
- OCR 噪声仍可能进入候选库，需要质量评分和评测持续优化。
- Qwen3 模型在 8-12GB GPU 上需要控制 batch size。
- FAISS + SQLite 适合当前阶段，后续资料规模大时可迁移到 Qdrant/Milvus。

约束：

- 所有用户可见输出必须保留知识来源引用。
- candidate 资料不能被包装成确定性权威依据。
- 安全/禁忌证据优先级高于普通相关性排序。
- 知识卡片必须可追溯到原始片段。
- 主系统不得静默降级到旧 RAG。

## 16. 交付物

实施完成后应交付：

- 独立 `NarrCare-KB` 服务代码；
- API 文档；
- SQLite schema；
- FAISS 索引构建脚本；
- 文件入库脚本；
- 自动扩充脚本；
- 调试界面；
- 评测集样例；
- `/eval/run` 指标报告；
- NarrCare 主系统集成代码；
- 部署说明，包含 Linux GPU 环境、模型下载、环境变量和启动命令。
