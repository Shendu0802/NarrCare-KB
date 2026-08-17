# NarrCare-KB 独立知识库服务 — 架构设计文档

## 项目概述

NarrCare-KB 是 NarrCare 项目的独立知识库服务，面向安宁疗护、死亡焦虑、叙事护理、家属沟通和护理干预场景，提供高质量、可解释、可追溯的知识检索与证据组织能力。

### 技术栈

| 层级 | 选型 |
|------|------|
| 后端框架 | Python 3.11+ / FastAPI / Pydantic v2 / Uvicorn |
| 向量检索 | FAISS (dense) + SQLite FTS5 (sparse) |
| Embedding | Qwen3-Embedding-0.6B（默认），BGE-M3（备选） |
| Reranker | Qwen3-Reranker-0.6B（默认），BGE-Reranker-v2-m3（备选） |
| LLM | 外部 OpenAI-compatible API（通用兼容层，不绑定厂商） |
| 文档解析 | PyMuPDF + PaddleOCR |
| 部署 | Linux GPU 服务器 (8-12GB 显存)，Docker 化 |

---

## Part 1：项目目录结构

```
NarrCare_KB/
├── models/                          # 已下载的模型文件
│   ├── Qwen3-Embedding-0.6B/
│   └── Qwen3-Reranker-0.6B/
├── 参考资料/                         # 待导入的原始资料
├── data/                            # 运行时数据
│   ├── db/                          # SQLite 数据库文件
│   ├── index/                       # FAISS 索引文件
│   └── eval/                        # 评测集
│       └── rag_queries.jsonl
├── src/                             # 源码根目录
│   ├── __init__.py
│   ├── config.py                    # 全局配置（模型路径、API keys、权重等）
│   ├── main.py                      # FastAPI app 入口
│   ├── models/                      # 共享数据模型（Pydantic）
│   │   ├── __init__.py
│   │   ├── knowledge_unit.py        # KnowledgeUnit schema
│   │   ├── evidence_bundle.py       # EvidenceBundle + EvidenceItem
│   │   ├── retrieval.py             # RetrieveRequest / RetrieveResponse
│   │   └── ingestion.py             # IngestRequest / IngestResponse
│   ├── ingestion/                   # 模块1：文档入库
│   │   ├── __init__.py
│   │   ├── api.py                   # POST /ingest/files, /ingest/search
│   │   ├── parser.py                # PDF/Markdown/Docx/JSONL 解析
│   │   ├── cleaner.py               # 文本清洗、去噪、质量评分
│   │   ├── chunker.py               # 语义切块
│   │   ├── enricher.py              # LLM 摘要/标签/知识卡片生成
│   │   └── orchestrator.py          # 入库 pipeline 编排
│   ├── retrieval/                   # 模块2：知识检索
│   │   ├── __init__.py
│   │   ├── api.py                   # POST /retrieve
│   │   ├── query_understanding.py   # LLM query analysis + rewrite
│   │   ├── dense.py                 # FAISS dense recall
│   │   ├── sparse.py                # BM25/FTS5 sparse recall
│   │   ├── metadata_recall.py       # 元数据/规则召回
│   │   ├── safety_recall.py         # 安全/禁忌强制召回
│   │   ├── hybrid.py                # 多路融合 + 权重
│   │   ├── reranker.py              # Qwen3 Reranker 重排
│   │   └── bundler.py               # 组装 EvidenceBundle
│   ├── indexing/                    # 模块3：索引管理
│   │   ├── __init__.py
│   │   ├── api.py                   # POST /index/rebuild
│   │   ├── vector_index.py          # FAISS 索引构建/加载
│   │   ├── text_index.py            # SQLite FTS5 索引
│   │   └── embedder.py              # Qwen3 Embedding 批量向量化
│   ├── evaluation/                  # 模块4：评测
│   │   ├── __init__.py
│   │   ├── api.py                   # POST /eval/run
│   │   └── metrics.py               # Recall@K, 安全命中率等指标
│   ├── debug/                       # 模块5：调试界面
│   │   ├── __init__.py
│   │   ├── api.py                   # GET /debug/status, /debug/query
│   │   └── templates/               # 简易 HTML 调试页面
│   ├── llm/                         # 共享：LLM 调用层
│   │   ├── __init__.py
│   │   └── client.py                # OpenAI-compatible API 客户端
│   ├── db/                          # 共享：数据库层
│   │   ├── __init__.py
│   │   ├── connection.py            # SQLite 连接管理
│   │   └── repository.py            # CRUD 操作封装
│   └── health/                      # 健康检查
│       └── api.py                   # GET /health
├── tests/                           # 测试
│   ├── test_ingestion/
│   ├── test_retrieval/
│   ├── test_indexing/
│   └── test_evaluation/
├── scripts/                         # 运维脚本
│   ├── download_models.sh
│   └── import_local_files.py
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .env.example
└── NarrCare_KB_Spec.md
```

### 依赖关系

```
main.py
  ├── config.py (全局配置)
  ├── models/ (共享 Pydantic schema，所有模块依赖)
  ├── db/ (共享数据库层)
  ├── llm/ (共享 LLM 客户端)
  └── 功能模块 (相互独立，不互相依赖):
      ├── health/api.py        → GET /health
      ├── ingestion/api.py     → POST /ingest/*
      ├── retrieval/api.py     → POST /retrieve
      ├── indexing/api.py      → POST /index/rebuild
      ├── evaluation/api.py    → POST /eval/run
      └── debug/api.py         → GET /debug/*
```

---

## Part 2：核心数据模型与数据库 Schema

### 2.1 SQLite 表设计

```sql
-- 知识单元主表
CREATE TABLE knowledge_units (
    id              TEXT PRIMARY KEY,          -- ku_xxx
    unit_type       TEXT NOT NULL,             -- semantic_chunk | knowledge_card
    source_type     TEXT NOT NULL,             -- pdf_book | paper | guideline | case | structured_library | markdown | web_reference
    source_status   TEXT NOT NULL DEFAULT 'candidate',  -- main | candidate | quarantined

    title           TEXT DEFAULT '',
    text            TEXT NOT NULL,
    summary         TEXT DEFAULT '',

    -- 来源追溯
    source_uri      TEXT DEFAULT '',
    source_title    TEXT DEFAULT '',
    source_citation TEXT DEFAULT '',
    page_start      INTEGER,
    page_end        INTEGER,

    -- 多维标签 (JSON 数组，SQLite 不支持原生数组)
    semantic_tags   TEXT DEFAULT '[]',
    scenario_tags   TEXT DEFAULT '[]',
    role_tags       TEXT DEFAULT '[]',
    method_tags     TEXT DEFAULT '[]',
    risk_levels     TEXT DEFAULT '[]',
    card_targets    TEXT DEFAULT '[]',
    contraindications TEXT DEFAULT '[]',

    -- 质量
    quality_score   REAL NOT NULL DEFAULT 0.0,
    quality_flags   TEXT DEFAULT '[]',         -- JSON: ["too_short", "copyright_page", ...]
    review_status   TEXT NOT NULL DEFAULT 'unreviewed',  -- unreviewed | approved | rejected

    -- 知识卡片关联
    parent_chunk_ids TEXT DEFAULT '[]',

    -- 解析元数据
    parse_method    TEXT DEFAULT '',            -- text_layer | ocr | hybrid
    embedding_model TEXT DEFAULT '',

    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- FTS5 全文索引（用于 sparse recall）
CREATE VIRTUAL TABLE knowledge_units_fts USING fts5(
    title, text, summary,
    content='knowledge_units',
    content_rowid='rowid'
);

-- 来源文档注册表（去重跟踪）
CREATE TABLE source_documents (
    id              TEXT PRIMARY KEY,
    file_path       TEXT,
    source_uri      TEXT,
    source_type     TEXT NOT NULL,
    title           TEXT DEFAULT '',
    total_pages     INTEGER,
    parse_status    TEXT NOT NULL DEFAULT 'pending',  -- pending | parsing | done | failed
    chunk_count     INTEGER DEFAULT 0,
    imported_at     TEXT NOT NULL
);

-- 评测运行记录
CREATE TABLE eval_runs (
    id              TEXT PRIMARY KEY,
    run_at          TEXT NOT NULL,
    total_queries   INTEGER,
    recall_at_10    REAL,
    safety_hit_rate REAL,
    noise_rate      REAL,
    avg_usability   REAL,
    details_json    TEXT DEFAULT '{}'
);
```

### 2.2 Pydantic Schema

```python
from pydantic import BaseModel
from typing import Literal, Optional

SourceType = Literal[
    "pdf_book", "paper", "guideline", "case",
    "structured_library", "markdown", "web_reference"
]
SourceStatus = Literal["main", "candidate", "quarantined"]
UnitType = Literal["semantic_chunk", "knowledge_card"]
ReviewStatus = Literal["unreviewed", "approved", "rejected"]
CardTarget = Literal["mindfulness", "healing", "communication", "personalized"]

class KnowledgeUnit(BaseModel):
    id: str
    unit_type: UnitType
    source_type: SourceType
    source_status: SourceStatus = "candidate"
    title: str = ""
    text: str
    summary: str = ""
    source_uri: str = ""
    source_title: str = ""
    source_citation: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    semantic_tags: list[str] = []
    scenario_tags: list[str] = []
    role_tags: list[str] = []
    method_tags: list[str] = []
    risk_levels: list[str] = []
    card_targets: list[CardTarget] = []
    contraindications: list[str] = []
    quality_score: float = 0.0
    review_status: ReviewStatus = "unreviewed"
    parent_chunk_ids: list[str] = []
    embedding_model: str = ""
    created_at: str
    updated_at: str

class EvidenceItem(BaseModel):
    """检索返回的单条证据"""
    id: str
    unit_type: str
    source_type: str
    source_status: str
    title: str
    snippet: str
    summary: str
    score: float
    source_citation: str
    source_uri: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    semantic_tags: list[str] = []
    scenario_tags: list[str] = []
    role_tags: list[str] = []
    method_tags: list[str] = []
    risk_levels: list[str] = []
    card_targets: list[str] = []
    quality_score: float = 0.0
    review_status: str = ""
    parent_chunk_ids: list[str] = []

class QueryAnalysis(BaseModel):
    intents: list[str] = []
    scenario_tags: list[str] = []
    role_focus: list[str] = []
    risk_signals: list[str] = []
    contraindication_signals: list[str] = []
    rewritten_queries: list[str] = []

class CardSources(BaseModel):
    mindfulness: list[EvidenceItem] = []
    healing: list[EvidenceItem] = []
    communication: list[EvidenceItem] = []
    personalized: list[EvidenceItem] = []

class RetrievalDebug(BaseModel):
    dense_hits: list = []
    sparse_hits: list = []
    metadata_hits: list = []
    reranked_hits: list = []
    model_versions: dict = {}
    index_version: str = ""

class EvidenceBundle(BaseModel):
    query_analysis: QueryAnalysis
    card_sources: CardSources
    supporting_passages: list[EvidenceItem]
    safety_and_boundary_evidence: list[EvidenceItem]
    candidate_evidence: list[EvidenceItem]
    excluded_evidence: list[EvidenceItem]
    retrieval_debug: Optional[RetrievalDebug] = None

class RetrieveRequest(BaseModel):
    session_id: str = ""
    patient_text: str = ""
    family_text: str = ""
    user_role: Literal["patient", "family", "nurse"] = "nurse"
    risk_assessment: dict = {}
    dyadic_analysis: dict = {}
    top_k_cards: int = 3
    top_k_passages: int = 5
    include_debug: bool = False

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    gpu_available: bool
    embedding_model: str
    reranker_model: str
    index_loaded: bool
    index_version: str
    document_count: int
    chunk_count: int
    card_count: int
```

### 2.3 ID 生成规范

```
格式: ku_{source_type_short}_{short_uuid}
示例: ku_pdf_book_a1b2c3d4, ku_paper_e5f6g7h8
```

---

## Part 3：入库 Pipeline 详细设计

### 3.1 整体流程

```
source_acquisition
  → document_parsing      (parser.py)
  → text_cleanup          (cleaner.py)
  → quality_scoring       (cleaner.py)
  → semantic_chunking     (chunker.py)
  → llm_enrichment        (enricher.py)
  → embedding             (indexing/embedder.py)
  → index_update          (indexing/vector_index.py + text_index.py)
```

### 3.2 资料获取 (source_acquisition)

支持两种入口：

**手动导入 (`POST /ingest/files`)：**
```python
class IngestFilesRequest(BaseModel):
    file_paths: list[str]          # 本地文件路径
    source_type: Optional[str] = None  # 不指定则自动推断
    source_status: str = "main"    # 手动导入默认为 main
```

**自动扩充 (`POST /ingest/search`)：**
```python
class IngestSearchRequest(BaseModel):
    topic: str
    keywords: list[str] = []
    source_priority: list[str] = ["pubmed", "pmc", "who", "guideline"]
    max_results: int = 20
```

### 3.3 文档解析 (parser.py)

```python
class DocumentParser:
    """支持 PDF、Markdown、Docx、JSONL 多种格式"""
    
    def parse(file_path: str) -> ParsedDocument:
        """返回 ParsedDocument，包含 raw_pages: list[Page]"""
    
class Page:
    page_number: int
    text: str          # 抽取的原始文本
    parse_method: str  # text_layer | ocr | hybrid
    confidence: float  # 文本层抽取 = 1.0，OCR = 识别置信度
```

解析策略：
- **PDF**: 先用 PyMuPDF 提取文本层；若文本量 < 阈值（如 50 字符/页），fallback 到 PaddleOCR
- **扫描版 PDF**: 全页 OCR，标记 `parse_method=ocr` 和较低的基础 confidence
- **Markdown/Docx**: 直接读取，标记 `parse_method=text_layer`，confidence=1.0
- **JSONL**: 按行解析，每行直接映射为结构化字段
- 解析结果先写入临时存储，不直接入库

### 3.4 文本清洗 (cleaner.py)

```python
class TextCleaner:
    def clean(self, page: Page) -> CleanedPage:
        """去页眉页脚、去噪、质量评分"""
    
    def score_quality(self, text: str) -> tuple[float, list[str]]:
        """返回 (quality_score, quality_flags)"""
```

清洗规则：
1. **页眉页脚去除**: 基于位置和重复模式识别
2. **目录页检测**: 关键词（"目录"、"Contents"）+ 点线模式
3. **版权页检测**: ISBN、CIP、版权符号、出版社信息聚集
4. **孤立字符过滤**: 连续中文字 < 5 且无完整句子
5. **乱码检测**: 非常见 Unicode 字符占比 > 30%
6. **低信息密度**: 标点/空白占比 > 70%
7. **重复去重**: 相邻页间相似度 > 0.9 则保留一份

质量评分规则：
```
quality_score = base_score - penalty_per_flag
base_score = 1.0 (text_layer) | 0.7 (ocr)
penalties:
  too_short: -0.3
  table_of_contents: -0.9 (直接隔离)
  copyright_page: -0.9 (直接隔离)
  isolated_characters: -0.4
  possible_ocr_garble: -0.3
  duplicated: -0.2
  low_information_density: -0.3
```

入库阈值:
- quality_score >= 0.5 → `source_status=main` (手动) 或 `candidate` (自动)
- quality_score < 0.5 → `source_status=quarantined`
- 目录页/版权页 → 直接 `quarantined`

### 3.5 语义切块 (chunker.py)

```python
class SemanticChunker:
    def chunk(self, document: CleanedDocument) -> list[Chunk]:
        """按语义边界切分，非固定字符数硬切"""
```

切块策略：
1. **边界检测优先级**: 标题/章节标记 > 自然段（双换行）> 单换行 > 句子结束
2. **目标大小**: 300-800 中文字符（约 150-400 个 token）
3. **重叠**: 相邻 chunk 间 80-150 中文字符 overlap
4. **过短合并**: < 200 字符的段落向前合并
5. **过长拆分**: > 1000 字符的段落按句子边界拆分
6. **保留元数据**: 每个 chunk 保留页码、原始位置、来源文档

### 3.6 LLM 入库增强 (enricher.py)

```python
class LLMEnricher:
    def enrich_chunk(self, chunk: Chunk) -> EnrichedChunk:
        """为 chunk 生成摘要和标签"""
    
    def generate_card(self, chunks: list[Chunk]) -> KnowledgeCard | None:
        """基于相关 chunk 生成知识卡片"""
```

对每个 semantic_chunk 调用 LLM 生成：
- `summary`: 200 字以内的中文摘要
- `semantic_tags`: 从预定义标签池中选择 (死亡焦虑、未竟事务、照护负担 等)
- `scenario_tags`: 适用于哪些护理场景
- `role_tags`: 适用的角色 (patient/family/nurse)
- `method_tags`: 涉及的方法 (正念、叙事疗法、尊严疗法 等)
- `risk_levels`: 风险等级
- `card_targets`: 映射到哪个卡片类型
- 可选: `knowledge_card`（当 chunk 信息足以支撑一个护理建议时）

LLM 输出验证：
- 标签必须在预定义标签池内（不在的丢弃或映射到最近标签）
- knowledge_card 必须包含 `parent_chunk_ids`
- 失败时 chunk 仍入库，只是标记为缺少自动标签

### 3.7 入库编排 (orchestrator.py)

```python
class IngestionOrchestrator:
    def ingest_file(self, file_path: str, source_type: str = None, source_status: str = "main") -> IngestionResult:
        """编排单个文件的完整入库流程"""
    
    def ingest_batch(self, file_paths: list[str]) -> list[IngestionResult]:
        """批量入库"""
```

编排逻辑：
1. 检查 `source_documents` 表是否已导入（基于 file_path 哈希去重）
2. 调用 Parser → Cleaner → Chunker → Enricher 四步
3. 将 chunk 写入 `knowledge_units` 表
4. 将 knowledge_card 写入 `knowledge_units` 表（带 parent_chunk_ids）
5. 记录到 `source_documents` 表
6. 返回入库统计（chunk 数、card 数、quarantined 数）

---

## Part 4：检索 Pipeline 详细设计

### 4.1 整体流程

```
POST /retrieve
  → query_understanding   (LLM 改写)
  → dense_recall          (FAISS)
  → sparse_recall         (FTS5/BM25)
  → metadata_recall       (标签/角色/卡片匹配)
  → safety_recall         (规则强制召回)
  → hybrid_fusion         (多路融合 + 权重)
  → rerank                (Qwen3 Reranker)
  → bundle_assemble       (组装 EvidenceBundle)
```

### 4.2 Query Understanding (query_understanding.py)

输入原始 patient_text + family_text + risk_assessment + dyadic_analysis，调用 LLM 生成：

```python
class QueryUnderstandingResult:
    intents: list[str]                 # ["死亡焦虑缓解", "家属沟通建议"]
    scenario_tags: list[str]            # ["夜间焦虑", "告别沟通"]
    role_focus: list[str]               # ["patient", "family"]
    risk_signals: list[str]             # ["绝望表达", "治疗拒绝"]
    contraindication_signals: list[str] # ["呼吸困难"]
    rewritten_queries: list[str]        # 至少4个变体:
        # 1. 原始表达保留版
        # 2. 场景化查询
        # 3. 护理干预查询
        # 4. 安全/禁忌查询
```

### 4.3 多路召回

**Dense Recall (dense.py):**
```python
class DenseRecaller:
    def recall(self, queries: list[str], top_k: int = 50) -> list[ScoredHit]:
        """每个 rewritten_query 分别查询 FAISS，合并去重"""
```
- 使用 Qwen3-Embedding-0.6B 对 rewritten_queries 向量化
- FAISS 搜索 top_k=50（后续 fusion 阶段再缩减）
- 返回带 cosine similarity 分数的 hit 列表

**Sparse Recall (sparse.py):**
```python
class SparseRecaller:
    def recall(self, queries: list[str], top_k: int = 30) -> list[ScoredHit]:
        """FTS5 全文搜索 + 简单 BM25 评分"""
```
- SQLite FTS5 对 title、text、summary 做全文匹配
- 简单 BM25 评分（基于词频和文档频率）
- 对中文做 jieba 分词后再查询

**Metadata Recall (metadata_recall.py):**
```python
class MetadataRecaller:
    def recall(self, query_analysis, top_k: int = 20) -> list[ScoredHit]:
        """基于标签、角色、卡片目标的精确匹配"""
```
- SQL 条件查询：`WHERE scenario_tags LIKE '%夜间焦虑%' AND role_tags LIKE '%nurse%'`
- 当前场景 + 角色 + 风险等级的交叉过滤
- card_targets 匹配（用于四卡片来源筛选）

**Safety Recall (safety_recall.py):**
```python
class SafetyRecaller:
    def recall(self, contraindication_signals: list[str]) -> list[ScoredHit]:
        """规则驱动的强制召回，不依赖分数"""
```
- 预定义禁忌词 → 知识单元的映射规则
- 例如 "呼吸困难" → 召回所有禁忌相关的 chunk
- 高风险表达（绝望、自杀意念）→ 强制召回安全边界证据
- 此路召回的结果标记 `safety_priority_boost`

### 4.4 多路融合 (hybrid.py)

```python
class HybridFusion:
    def fuse(self, dense_hits, sparse_hits, metadata_hits, safety_hits) -> list[ScoredHit]:
        """多路去重 + 加权融合"""
```

融合权重（默认值，可在 config.py 调整）：
```
dense/sparse recall score: 0.20
rerank_score (此阶段为 0，待 reranker 填充): 0.55
metadata match: 0.10
quality score: 0.10
source status: 0.05
```

### 4.5 Reranker (reranker.py)

```python
class Qwen3Reranker:
    def rerank(self, query: str, candidates: list[KnowledgeUnit]) -> list[ScoredHit]:
        """使用 Qwen3-Reranker-0.6B 对候选重排"""
```

重排输入（拼接为 prompt）：
- 用户原始文本 + query analysis
- 候选知识文本 + 标签 + 来源状态 + 质量分

最终分数计算：
```
final_score = rerank_score * 0.55
            + recall_score * 0.20
            + metadata_match * 0.10
            + quality_score * 0.10
            + source_status_weight * 0.05
            + safety_priority_boost (加法叠加)
```

### 4.6 证据包组装 (bundler.py)

```python
class Bundler:
    def assemble(self, reranked: list[ScoredHit], query_analysis, top_k_cards, top_k_passages) -> EvidenceBundle:
        """按层次分类组装"""
```

分类规则：
1. **card_sources**: 按 card_targets 分入 mindfulness/healing/communication/personalized 四组，每组取 top_k_cards 条
2. **supporting_passages**: source_status=main 且非卡片的优质 semantic_chunk，取 top_k_passages 条
3. **safety_and_boundary_evidence**: 带 safety_priority_boost 标记的 + 高风险/禁忌标签的
4. **candidate_evidence**: source_status=candidate，分数达标但权重打折
5. **excluded_evidence**: source_status=quarantined，只进入 debug

---

## Part 5：索引管理设计

### 5.1 Embedding 生成 (embedder.py)

```python
class Embedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = load_qwen3_embedding(model_path, device)
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """批量文本向量化，返回 (N, dim) numpy array"""
    
    def encode_query(self, query: str) -> np.ndarray:
        """单条查询向量化"""
```

设计要点：
- 模型加载时使用 FP16 以节省显存
- batch_size 默认 32（适配 8-12GB 显存限制）
- 入库时批量处理，单条检索时实时编码
- 向量维度取决于模型：Qwen3-Embedding-0.6B = 1024 维

### 5.2 FAISS 索引 (vector_index.py)

```python
class VectorIndex:
    def __init__(self, dim: int = 1024, index_path: str = "data/index/faiss.index"):
        self.dim = dim
        self.index_path = index_path
    
    def build(self, chunks: list[KnowledgeUnit], embedder: Embedder) -> None:
        """全量构建 FAISS 索引"""
        # 1. 遍历所有 main + candidate chunk
        # 2. 批量生成 embedding
        # 3. 构建 IndexFlatIP (内积搜索，等价于 cosine)
        # 4. 可选: 添加 IVF 索引加速（数据量 > 10k 时）
        # 5. 持久化到磁盘
    
    def load(self) -> bool:
        """加载已有索引，返回是否成功"""
    
    def search(self, query_vector: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        """返回 (chunk_db_id, score) 列表"""
    
    def add(self, new_chunks: list[KnowledgeUnit], embedder: Embedder) -> None:
        """增量添加（Phase 2+）"""
```

索引存储结构：
- `data/index/faiss.index` — FAISS 索引文件
- `data/index/id_map.json` — FAISS 内部 ID → knowledge_unit ID 映射
- `data/index/meta.json` — 索引元数据（版本、向量数、维度、模型名、构建时间）

### 5.3 FTS5 文本索引 (text_index.py)

```python
class TextIndex:
    def __init__(self, db_path: str = "data/db/kb.sqlite"):
        self.db_path = db_path
    
    def build(self, chunks: list[KnowledgeUnit]) -> None:
        """重建 FTS5 全文索引"""
        # INSERT INTO knowledge_units_fts(rowid, title, text, summary) VALUES (...)
    
    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        """FTS5 全文搜索，返回 (ku_id, bm25_score)"""
```

### 5.4 索引重建 (POST /index/rebuild)

```python
class IndexRebuildOrchestrator:
    def rebuild(self, full: bool = True) -> RebuildResult:
        """全量重建（Phase 1 不支持增量）"""
        # 1. 从 SQLite 加载所有 main + candidate chunk
        # 2. 调用 Embedder 批量向量化
        # 3. 重建 FAISS 索引
        # 4. 重建 FTS5 索引
        # 5. 更新索引版本号和元数据
        # 6. 返回统计信息
```

---

## Part 6：LLM 客户端设计

### 6.1 OpenAI-Compatible 客户端 (llm/client.py)

```python
class LLMClient:
    """通用 OpenAI-compatible API 客户端，不绑定具体厂商"""
    
    def __init__(self, config: LLMConfig):
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.default_model = config.model
        self.timeout = config.timeout
    
    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: dict | None = None,  # JSON mode
    ) -> ChatResponse:
        """通用 chat completion 调用"""
    
    async def chat_with_json(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> dict:
        """chat + JSON mode，解析为 dict，失败重试一次"""
```

### 6.2 配置

```python
class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com/v1"    # 默认，可覆盖
    api_key: str = ""                                  # 从环境变量 LLM_API_KEY 读取
    model: str = "deepseek-chat"
    timeout: int = 60
    max_retries: int = 2
```

### 6.3 调用场景

| 场景 | 模块 | temperature | max_tokens | JSON mode |
|------|------|-------------|------------|-----------|
| Chunk 标签增强 | enricher.py | 0.3 | 1024 | ✅ |
| Knowledge Card 生成 | enricher.py | 0.5 | 2048 | ✅ |
| Query Understanding | query_understanding.py | 0.3 | 1024 | ✅ |
| 自动扩充搜索词生成 | ingestion/api.py | 0.5 | 512 | ✅ |

---

## Part 7：配置管理

### 7.1 config.py 设计

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 服务
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    
    # 路径
    models_dir: str = "models"
    data_dir: str = "data"
    db_path: str = "data/db/kb.sqlite"
    faiss_index_path: str = "data/index/faiss.index"
    
    # 模型
    embedding_model: str = "Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen3-Reranker-0.6B"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    
    # LLM
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60
    
    # 检索
    dense_top_k: int = 50
    sparse_top_k: int = 30
    metadata_top_k: int = 20
    safety_top_k: int = 10
    default_top_k_cards: int = 3
    default_top_k_passages: int = 5
    
    # 权重
    weight_rerank: float = 0.55
    weight_recall: float = 0.20
    weight_metadata: float = 0.10
    weight_quality: float = 0.10
    weight_source_status: float = 0.05
    
    # Chunking
    chunk_min_chars: int = 300
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100
    
    # 质量阈值
    quality_main_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        env_prefix = "KB_"
```

---

## Part 8：API 路由与错误处理

### 8.1 路由注册

```python
# src/main.py
from fastapi import FastAPI
from src.health.api import router as health_router
from src.retrieval.api import router as retrieval_router
from src.ingestion.api import router as ingestion_router
from src.indexing.api import router as indexing_router
from src.evaluation.api import router as eval_router
from src.debug.api import router as debug_router

app = FastAPI(title="NarrCare-KB", version="0.1.0")

app.include_router(health_router)       # GET  /health
app.include_router(retrieval_router)     # POST /retrieve
app.include_router(ingestion_router)     # POST /ingest/files, /ingest/search
app.include_router(indexing_router)      # POST /index/rebuild
app.include_router(eval_router)          # POST /eval/run
app.include_router(debug_router)         # GET  /debug/status, /debug/query
```

### 8.2 错误处理

```python
class KBErrorCode:
    KB_SERVICE_UNAVAILABLE = "KB_SERVICE_UNAVAILABLE"
    KB_INDEX_NOT_READY = "KB_INDEX_NOT_READY"
    KB_SCHEMA_INVALID = "KB_SCHEMA_INVALID"
    KB_RETRIEVAL_TIMEOUT = "KB_RETRIEVAL_TIMEOUT"
    KB_LLM_ERROR = "KB_LLM_ERROR"
    KB_EMBEDDING_ERROR = "KB_EMBEDDING_ERROR"
    KB_INGESTION_FAILED = "KB_INGESTION_FAILED"

# 全局异常处理器
@app.exception_handler(KBException)
async def kb_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.error_code, "detail": exc.detail}
    )
```

---

## Part 9：评测设计

### 9.1 评测集格式

```jsonl
{"query": "...", "patient_context": "...", "family_context": "...", "expected_tags": [...], "must_include_source_types": [...], "must_include_card_targets": [...], "must_include_ids": [...], "must_not_include_flags": [...], "risk_level": "...", "human_notes": "..."}
```

### 9.2 指标计算

- **Recall@10**: 期望 ID 在返回 top-10 中的比例
- **安全/禁忌命中率**: 有风险标记的 query 是否包含 safety_and_boundary_evidence
- **噪声率**: excluded_evidence 中 OCR/目录/乱码 chunk 占总返回的比例
- **人工可用性评分**: 人工标注的 1-5 分均值

### 9.3 评测流程

```
POST /eval/run
  → 加载 rag_queries.jsonl
  → 逐条调用 /retrieve
  → 计算指标
  → 记录到 eval_runs 表
  → 返回指标报告
```

---

## Part 10：调试界面

### 10.1 页面功能

简易 FastAPI HTML 调试界面，不做完整后台：

1. **检索测试页** (`/debug/query`): 输入框 + EvidenceBundle 完整展示
2. **状态页** (`/debug/status`): 索引状态、模型版本、统计图表
3. 不包含: 人工审核流、批量编辑、多用户权限

### 10.2 API

```python
GET /debug/status
# 返回: 详细状态信息，包括 source_type 分布、main/candidate/quarantined 计数

GET /debug/query?session_id=xxx&patient_text=xxx&family_text=xxx
# 返回: 完整 EvidenceBundle + 各阶段中间结果
```

---

## Part 11：实施阶段映射

每个 Phase 对应 Code 模块的实现：

| Phase | 内容 | 对应模块 |
|-------|------|----------|
| **P1** 服务骨架与数据模型 | `main.py`, `config.py`, `models/`, `db/`, `health/`, `llm/client.py` |
| **P2** 文档入库与清洗 | `ingestion/parser.py`, `cleaner.py`, `chunker.py`, `orchestrator.py`, `api.py` |
| **P3** LLM 标签与知识卡片 | `ingestion/enricher.py`（依赖 P2 的 chunk 输出） |
| **P4** GPU 向量化与索引 | `indexing/embedder.py`, `vector_index.py`, `text_index.py`, `api.py` |
| **P5** Query Understanding + Hybrid | `retrieval/query_understanding.py`, `dense.py`, `sparse.py`, `metadata_recall.py`, `safety_recall.py`, `hybrid.py` |
| **P6** Reranker + 分层证据包 | `retrieval/reranker.py`, `bundler.py` |
| **P7** 自动扩充与评测 | `ingestion/api.py`（/ingest/search）, `evaluation/` |

---

## 附录 A：配置文件示例 (.env.example)

```bash
# 服务
KB_HOST=0.0.0.0
KB_PORT=9000

# 模型路径
KB_MODELS_DIR=models
KB_EMBEDDING_MODEL=Qwen3-Embedding-0.6B
KB_RERANKER_MODEL=Qwen3-Reranker-0.6B
KB_EMBEDDING_DEVICE=cuda
KB_EMBEDDING_BATCH_SIZE=32

# LLM API (OpenAI-compatible)
KB_LLM_BASE_URL=https://api.deepseek.com/v1
KB_LLM_API_KEY=sk-your-key-here
KB_LLM_MODEL=deepseek-chat
KB_LLM_TIMEOUT=60

# 数据路径
KB_DATA_DIR=data
KB_DB_PATH=data/db/kb.sqlite
KB_FAISS_INDEX_PATH=data/index/faiss.index

# 检索默认值
KB_DENSE_TOP_K=50
KB_SPARSE_TOP_K=30
KB_DEFAULT_TOP_K_CARDS=3
KB_DEFAULT_TOP_K_PASSAGES=5

# 权重
KB_WEIGHT_RERANK=0.55
KB_WEIGHT_RECALL=0.20
KB_WEIGHT_METADATA=0.10
KB_WEIGHT_QUALITY=0.10
KB_WEIGHT_SOURCE_STATUS=0.05
```

## 附录 B：依赖清单 (requirements.txt)

```
# Web framework
fastapi==0.115.*
uvicorn[standard]==0.32.*
pydantic==2.*
pydantic-settings==2.*

# Database
aiosqlite==0.20.*

# ML
torch>=2.0
transformers>=4.45
faiss-cpu>=1.8   # 或 faiss-gpu
sentence-transformers>=3.0

# Document parsing
PyMuPDF>=1.24
paddleocr>=2.9
python-docx>=1.1
markdown>=3.7

# Text processing
jieba>=0.42

# LLM client
openai>=1.50
httpx>=0.27

# Evaluation
numpy>=1.26
scikit-learn>=1.5

# Debug UI (optional)
jinja2>=3.1
```
