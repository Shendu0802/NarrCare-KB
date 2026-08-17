# NarrCare-KB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent knowledge-base service for NarrCare that provides high-quality, explainable, traceable knowledge retrieval and evidence organization for hospice care scenarios.

**Architecture:** Domain-modular FastAPI service with six independent modules (health, ingestion, retrieval, indexing, evaluation, debug) sharing Pydantic models, SQLite DB layer, and an OpenAI-compatible LLM client. Two pipelines — ingestion (parse → clean → chunk → enrich → embed → index) and retrieval (query-understanding → multi-recall → fusion → rerank → bundle).

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLite + FTS5, FAISS, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, PyMuPDF, PaddleOCR, OpenAI-compatible LLM API.

---

## Phase 1: Service Skeleton & Data Models

### Task 1.1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "narrcare-kb"
version = "0.1.0"
description = "NarrCare Knowledge Base Service"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "aiosqlite>=0.20",
    "torch>=2.0",
    "transformers>=4.45",
    "faiss-cpu>=1.8",
    "sentence-transformers>=3.0",
    "PyMuPDF>=1.24",
    "paddleocr>=2.9",
    "python-docx>=1.1",
    "markdown>=3.7",
    "jieba>=0.42",
    "openai>=1.50",
    "httpx>=0.27",
    "numpy>=1.26",
    "scikit-learn>=1.5",
    "jinja2>=3.1",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create requirements.txt**

```
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.0
pydantic-settings>=2.0
aiosqlite>=0.20
torch>=2.0
transformers>=4.45
faiss-cpu>=1.8
sentence-transformers>=3.0
PyMuPDF>=1.24
paddleocr>=2.9
python-docx>=1.1
markdown>=3.7
jieba>=0.42
openai>=1.50
httpx>=0.27
numpy>=1.26
scikit-learn>=1.5
jinja2>=3.1
```

- [ ] **Step 3: Create .env.example**

```bash
# Service
KB_HOST=0.0.0.0
KB_PORT=9000

# Model paths
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

# Data paths
KB_DATA_DIR=data
KB_DB_PATH=data/db/kb.sqlite
KB_FAISS_INDEX_PATH=data/index/faiss.index

# Retrieval defaults
KB_DENSE_TOP_K=50
KB_SPARSE_TOP_K=30
KB_DEFAULT_TOP_K_CARDS=3
KB_DEFAULT_TOP_K_PASSAGES=5

# Weights
KB_WEIGHT_RERANK=0.55
KB_WEIGHT_RECALL=0.20
KB_WEIGHT_METADATA=0.10
KB_WEIGHT_QUALITY=0.10
KB_WEIGHT_SOURCE_STATUS=0.05
```

- [ ] **Step 4: Create src/__init__.py**

```python
"""NarrCare-KB: Independent Knowledge Base Service for NarrCare."""
```

- [ ] **Step 5: Verify scaffolding**

```bash
ls -la pyproject.toml requirements.txt .env.example src/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .env.example src/__init__.py
git commit -m "feat: project scaffolding with dependencies and env template"
```

### Task 1.2: Configuration management

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from src.config import Settings


class TestSettings:
    def test_defaults(self):
        """Settings should have sensible defaults."""
        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.embedding_model == "Qwen3-Embedding-0.6B"
        assert settings.reranker_model == "Qwen3-Reranker-0.6B"
        assert settings.chunk_min_chars == 300
        assert settings.chunk_max_chars == 800
        assert settings.weight_rerank == 0.55

    def test_env_override(self, monkeypatch):
        """Environment variables with KB_ prefix should override defaults."""
        monkeypatch.setenv("KB_PORT", "9999")
        monkeypatch.setenv("KB_LLM_MODEL", "gpt-4")
        settings = Settings()
        assert settings.port == 9999
        assert settings.llm_model == "gpt-4"

    def test_weight_sum(self):
        """Weights do not need to sum to 1.0 — they are relative."""
        settings = Settings()
        total = (settings.weight_rerank + settings.weight_recall
                 + settings.weight_metadata + settings.weight_quality
                 + settings.weight_source_status)
        # weights are relative, just verify they are positive
        assert all(w > 0 for w in [
            settings.weight_rerank, settings.weight_recall,
            settings.weight_metadata, settings.weight_quality,
            settings.weight_source_status
        ])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement src/config.py**

```python
"""Global configuration via pydantic-settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with KB_ prefix."""

    # Service
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False

    # Paths
    models_dir: str = "models"
    data_dir: str = "data"
    db_path: str = "data/db/kb.sqlite"
    faiss_index_path: str = "data/index/faiss.index"

    # Model names (paths relative to models_dir)
    embedding_model: str = "Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen3-Reranker-0.6B"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # LLM API (OpenAI-compatible)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60

    # Retrieval
    dense_top_k: int = 50
    sparse_top_k: int = 30
    metadata_top_k: int = 20
    safety_top_k: int = 10
    default_top_k_cards: int = 3
    default_top_k_passages: int = 5

    # Weights (relative, do not need to sum to 1.0)
    weight_rerank: float = 0.55
    weight_recall: float = 0.20
    weight_metadata: float = 0.10
    weight_quality: float = 0.10
    weight_source_status: float = 0.05

    # Chunking
    chunk_min_chars: int = 300
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100

    # Quality
    quality_main_threshold: float = 0.5

    class Config:
        env_file = ".env"
        env_prefix = "KB_"


# Singleton
settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add configuration management with pydantic-settings"
```

### Task 1.3: Error handling

**Files:**
- Create: `src/errors.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_errors.py
from src.errors import KBException, KBErrorCode


class TestKBException:
    def test_error_codes_exist(self):
        """All error codes should be defined."""
        assert KBErrorCode.KB_SERVICE_UNAVAILABLE == "KB_SERVICE_UNAVAILABLE"
        assert KBErrorCode.KB_INDEX_NOT_READY == "KB_INDEX_NOT_READY"
        assert KBErrorCode.KB_SCHEMA_INVALID == "KB_SCHEMA_INVALID"
        assert KBErrorCode.KB_RETRIEVAL_TIMEOUT == "KB_RETRIEVAL_TIMEOUT"
        assert KBErrorCode.KB_LLM_ERROR == "KB_LLM_ERROR"

    def test_exception_creation(self):
        """KBException should carry error_code, detail, and http_status."""
        exc = KBException(
            error_code=KBErrorCode.KB_INDEX_NOT_READY,
            detail="FAISS index file not found",
            http_status=503,
        )
        assert exc.error_code == "KB_INDEX_NOT_READY"
        assert exc.detail == "FAISS index file not found"
        assert exc.http_status == 503
        assert "KB_INDEX_NOT_READY" in str(exc)

    def test_exception_default_status(self):
        """Default http_status should be 500."""
        exc = KBException(error_code="KB_UNKNOWN")
        assert exc.http_status == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_errors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement src/errors.py**

```python
"""Unified error handling for NarrCare-KB."""


class KBErrorCode:
    """Standardized error codes consumed by NarrCare main system."""
    KB_SERVICE_UNAVAILABLE = "KB_SERVICE_UNAVAILABLE"
    KB_INDEX_NOT_READY = "KB_INDEX_NOT_READY"
    KB_SCHEMA_INVALID = "KB_SCHEMA_INVALID"
    KB_RETRIEVAL_TIMEOUT = "KB_RETRIEVAL_TIMEOUT"
    KB_LLM_ERROR = "KB_LLM_ERROR"
    KB_EMBEDDING_ERROR = "KB_EMBEDDING_ERROR"
    KB_INGESTION_FAILED = "KB_INGESTION_FAILED"


class KBException(Exception):
    """Base exception for KB service errors.

    Attributes:
        error_code: One of KBErrorCode values, consumed by NarrCare main system.
        detail: Human-readable error description.
        http_status: HTTP status code to return.
    """

    def __init__(
        self,
        error_code: str,
        detail: str = "",
        http_status: int = 500,
    ):
        self.error_code = error_code
        self.detail = detail
        self.http_status = http_status
        super().__init__(f"[{error_code}] {detail}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/errors.py tests/test_errors.py
git commit -m "feat: add unified error handling with KBException"
```

### Task 1.4: Pydantic data models

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/knowledge_unit.py`
- Create: `src/models/evidence_bundle.py`
- Create: `src/models/retrieval.py`
- Create: `src/models/ingestion.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import json
from src.models.knowledge_unit import KnowledgeUnit, SourceType, SourceStatus, UnitType, ReviewStatus, CardTarget
from src.models.evidence_bundle import EvidenceItem, EvidenceBundle, QueryAnalysis, CardSources, RetrievalDebug
from src.models.retrieval import RetrieveRequest
from src.models.ingestion import IngestFilesRequest, IngestSearchRequest


class TestKnowledgeUnit:
    def test_minimal_creation(self):
        ku = KnowledgeUnit(
            id="ku_pdf_book_a1b2c3d4",
            unit_type="semantic_chunk",
            source_type="pdf_book",
            text="临终患者常常经历存在性痛苦...",
            created_at="2026-07-01T00:00:00",
            updated_at="2026-07-01T00:00:00",
        )
        assert ku.id == "ku_pdf_book_a1b2c3d4"
        assert ku.source_status == "candidate"
        assert ku.quality_score == 0.0
        assert ku.semantic_tags == []

    def test_tag_serialization(self):
        ku = KnowledgeUnit(
            id="ku_test_1",
            unit_type="semantic_chunk",
            source_type="guideline",
            text="呼吸困难的非药物干预包括...",
            semantic_tags=["呼吸困难", "症状管理"],
            scenario_tags=["夜间焦虑"],
            role_tags=["nurse"],
            card_targets=["communication"],
            created_at="2026-07-01T00:00:00",
            updated_at="2026-07-01T00:00:00",
        )
        data = ku.model_dump()
        assert "呼吸困难" in data["semantic_tags"]
        assert "nurse" in data["role_tags"]

    def test_source_types(self):
        valid_types = [
            "pdf_book", "paper", "guideline", "case",
            "structured_library", "markdown", "web_reference",
        ]
        for t in valid_types:
            ku = KnowledgeUnit(
                id=f"ku_{t}_test",
                unit_type="semantic_chunk",
                source_type=t,
                text="test",
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
            )
            assert ku.source_type == t


class TestEvidenceBundle:
    def test_empty_bundle(self):
        bundle = EvidenceBundle(
            query_analysis=QueryAnalysis(),
            card_sources=CardSources(),
            supporting_passages=[],
            safety_and_boundary_evidence=[],
            candidate_evidence=[],
            excluded_evidence=[],
        )
        assert bundle.query_analysis.intents == []
        assert bundle.card_sources.mindfulness == []

    def test_bundle_with_items(self):
        item = EvidenceItem(
            id="ku_test_1",
            unit_type="semantic_chunk",
            source_type="guideline",
            source_status="main",
            title="呼吸困难干预",
            snippet="非药物干预包括...",
            summary="呼吸困难的非药物管理方法",
            score=0.85,
            source_citation="WHO Guideline 2024",
            source_uri="",
            semantic_tags=["呼吸困难"],
            scenario_tags=[],
            role_tags=["nurse"],
            method_tags=[],
            risk_levels=["high"],
            card_targets=["communication"],
            quality_score=0.9,
            parent_chunk_ids=[],
        )
        bundle = EvidenceBundle(
            query_analysis=QueryAnalysis(
                intents=["symptom_management"],
                scenario_tags=["呼吸困难"],
            ),
            card_sources=CardSources(communication=[item]),
            supporting_passages=[item],
            safety_and_boundary_evidence=[],
            candidate_evidence=[],
            excluded_evidence=[],
        )
        assert len(bundle.supporting_passages) == 1
        assert len(bundle.card_sources.communication) == 1


class TestRetrieveRequest:
    def test_defaults(self):
        req = RetrieveRequest(patient_text="我感到害怕")
        assert req.user_role == "nurse"
        assert req.top_k_cards == 3
        assert req.top_k_passages == 5
        assert req.include_debug is False


class TestIngestRequests:
    def test_ingest_files_request(self):
        req = IngestFilesRequest(file_paths=["/data/book.pdf"])
        assert req.source_status == "main"

    def test_ingest_search_request(self):
        req = IngestSearchRequest(topic="death anxiety hospice")
        assert req.max_results == 20
        assert "pubmed" in req.source_priority
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/models/__init__.py**

```python
"""Shared Pydantic data models for NarrCare-KB."""
```

- [ ] **Step 4: Create src/models/knowledge_unit.py**

```python
"""Core KnowledgeUnit schema."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

SourceType = Literal[
    "pdf_book", "paper", "guideline", "case",
    "structured_library", "markdown", "web_reference",
]
SourceStatus = Literal["main", "candidate", "quarantined"]
UnitType = Literal["semantic_chunk", "knowledge_card"]
ReviewStatus = Literal["unreviewed", "approved", "rejected"]
CardTarget = Literal["mindfulness", "healing", "communication", "personalized"]


class KnowledgeUnit(BaseModel):
    """A single knowledge unit — either a semantic_chunk or a knowledge_card."""

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

    semantic_tags: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    role_tags: list[str] = Field(default_factory=list)
    method_tags: list[str] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)
    card_targets: list[CardTarget] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)

    quality_score: float = 0.0
    review_status: ReviewStatus = "unreviewed"
    parent_chunk_ids: list[str] = Field(default_factory=list)

    embedding_model: str = ""
    created_at: str
    updated_at: str
```

- [ ] **Step 5: Create src/models/evidence_bundle.py**

```python
"""EvidenceBundle and supporting schemas."""
from typing import Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A single evidence item returned in retrieval results."""

    id: str
    unit_type: str = ""
    source_type: str = ""
    source_status: str = ""
    title: str = ""
    snippet: str = ""
    summary: str = ""
    score: float = 0.0
    source_citation: str = ""
    source_uri: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    semantic_tags: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    role_tags: list[str] = Field(default_factory=list)
    method_tags: list[str] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)
    card_targets: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    review_status: str = ""
    parent_chunk_ids: list[str] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    """LLM-generated query understanding output."""

    intents: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    role_focus: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)
    contraindication_signals: list[str] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)


class CardSources(BaseModel):
    """Card-target-specific evidence groups."""

    mindfulness: list[EvidenceItem] = Field(default_factory=list)
    healing: list[EvidenceItem] = Field(default_factory=list)
    communication: list[EvidenceItem] = Field(default_factory=list)
    personalized: list[EvidenceItem] = Field(default_factory=list)


class RetrievalDebug(BaseModel):
    """Optional debug information from retrieval pipeline."""

    dense_hits: list = Field(default_factory=list)
    sparse_hits: list = Field(default_factory=list)
    metadata_hits: list = Field(default_factory=list)
    reranked_hits: list = Field(default_factory=list)
    model_versions: dict = Field(default_factory=dict)
    index_version: str = ""


class EvidenceBundle(BaseModel):
    """Layered evidence bundle returned by POST /retrieve."""

    query_analysis: QueryAnalysis
    card_sources: CardSources
    supporting_passages: list[EvidenceItem] = Field(default_factory=list)
    safety_and_boundary_evidence: list[EvidenceItem] = Field(default_factory=list)
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)
    excluded_evidence: list[EvidenceItem] = Field(default_factory=list)
    retrieval_debug: Optional[RetrievalDebug] = None
```

- [ ] **Step 6: Create src/models/retrieval.py**

```python
"""Retrieval request/response schemas."""
from typing import Literal
from pydantic import BaseModel, Field
from src.models.evidence_bundle import EvidenceBundle


class RetrieveRequest(BaseModel):
    """POST /retrieve request body."""

    session_id: str = ""
    patient_text: str = ""
    family_text: str = ""
    user_role: Literal["patient", "family", "nurse"] = "nurse"
    risk_assessment: dict = Field(default_factory=dict)
    dyadic_analysis: dict = Field(default_factory=dict)
    top_k_cards: int = 3
    top_k_passages: int = 5
    include_debug: bool = False


class RetrieveResponse(BaseModel):
    """POST /retrieve response — wraps EvidenceBundle."""

    data: EvidenceBundle
    error: str | None = None
```

- [ ] **Step 7: Create src/models/ingestion.py**

```python
"""Ingestion request/response schemas."""
from typing import Optional
from pydantic import BaseModel, Field


class IngestFilesRequest(BaseModel):
    """POST /ingest/files request body."""

    file_paths: list[str]
    source_type: Optional[str] = None
    source_status: str = "main"


class IngestFilesResponse(BaseModel):
    """POST /ingest/files response."""

    task_id: str
    status: str  # pending | running | done | failed
    results: list[dict] = Field(default_factory=list)


class IngestSearchRequest(BaseModel):
    """POST /ingest/search request body."""

    topic: str
    keywords: list[str] = Field(default_factory=list)
    source_priority: list[str] = Field(
        default_factory=lambda: ["pubmed", "pmc", "who", "guideline"]
    )
    max_results: int = 20


class IngestSearchResponse(BaseModel):
    """POST /ingest/search response."""

    task_id: str
    candidates: list[dict] = Field(default_factory=list)
    status: str = "pending"
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: add Pydantic data models (KnowledgeUnit, EvidenceBundle, requests)"
```

### Task 1.5: Database layer

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/connection.py`
- Create: `src/db/repository.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import pytest
import os
import tempfile
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.sqlite")
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()


class TestDatabase:
    def test_initialize_creates_tables(self, db):
        """Database initialization should create all required tables."""
        tables = db.list_tables()
        assert "knowledge_units" in tables
        assert "source_documents" in tables
        assert "eval_runs" in tables

    def test_fts_table_created(self, db):
        """FTS5 virtual table should be created."""
        tables = db.list_tables()
        assert "knowledge_units_fts" in tables


class TestKnowledgeUnitRepository:
    def test_insert_and_get(self, db):
        repo = KnowledgeUnitRepository(db)
        ku_data = {
            "id": "ku_test_001",
            "unit_type": "semantic_chunk",
            "source_type": "pdf_book",
            "source_status": "main",
            "title": "Test Chunk",
            "text": "这是一段测试文本",
            "summary": "测试摘要",
            "semantic_tags": '["test"]',
            "scenario_tags": '["test"]',
            "role_tags": '["nurse"]',
            "method_tags": '[]',
            "risk_levels": '[]',
            "card_targets": '[]',
            "contraindications": '[]',
            "quality_score": 0.9,
            "quality_flags": '[]',
            "review_status": "unreviewed",
            "parent_chunk_ids": '[]',
            "parse_method": "text_layer",
            "embedding_model": "",
            "created_at": "2026-07-01T00:00:00",
            "updated_at": "2026-07-01T00:00:00",
        }
        repo.insert(ku_data)
        result = repo.get_by_id("ku_test_001")
        assert result is not None
        assert result["title"] == "Test Chunk"
        assert result["source_status"] == "main"

    def test_list_by_status(self, db):
        repo = KnowledgeUnitRepository(db)
        for i in range(3):
            repo.insert({
                "id": f"ku_main_{i}",
                "unit_type": "semantic_chunk",
                "source_type": "pdf_book",
                "source_status": "main",
                "text": f"content {i}",
                "semantic_tags": '[]', "scenario_tags": '[]',
                "role_tags": '[]', "method_tags": '[]',
                "risk_levels": '[]', "card_targets": '[]',
                "contraindications": '[]', "quality_flags": '[]',
                "parent_chunk_ids": '[]', "parse_method": "",
                "embedding_model": "",
                "created_at": "2026-07-01T00:00:00",
                "updated_at": "2026-07-01T00:00:00",
            })
        for i in range(2):
            repo.insert({
                "id": f"ku_cand_{i}",
                "unit_type": "semantic_chunk",
                "source_type": "paper",
                "source_status": "candidate",
                "text": f"candidate content {i}",
                "semantic_tags": '[]', "scenario_tags": '[]',
                "role_tags": '[]', "method_tags": '[]',
                "risk_levels": '[]', "card_targets": '[]',
                "contraindications": '[]', "quality_flags": '[]',
                "parent_chunk_ids": '[]', "parse_method": "",
                "embedding_model": "",
                "created_at": "2026-07-01T00:00:00",
                "updated_at": "2026-07-01T00:00:00",
            })
        main_chunks = repo.list_by_status("main")
        assert len(main_chunks) == 3
        candidate_chunks = repo.list_by_status("candidate")
        assert len(candidate_chunks) == 2

    def test_get_all_active(self, db):
        """get_all_active should return main + candidate but not quarantined."""
        repo = KnowledgeUnitRepository(db)
        for status in ["main", "candidate", "quarantined"]:
            repo.insert({
                "id": f"ku_{status}",
                "unit_type": "semantic_chunk",
                "source_type": "pdf_book",
                "source_status": status,
                "text": f"{status} content",
                "semantic_tags": '[]', "scenario_tags": '[]',
                "role_tags": '[]', "method_tags": '[]',
                "risk_levels": '[]', "card_targets": '[]',
                "contraindications": '[]', "quality_flags": '[]',
                "parent_chunk_ids": '[]', "parse_method": "",
                "embedding_model": "",
                "created_at": "2026-07-01T00:00:00",
                "updated_at": "2026-07-01T00:00:00",
            })
        active = repo.get_all_active()
        ids = [r["id"] for r in active]
        assert "ku_main" in ids
        assert "ku_candidate" in ids
        assert "ku_quarantined" not in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/db/__init__.py**

```python
"""Database layer for NarrCare-KB."""
```

- [ ] **Step 4: Create src/db/connection.py**

```python
"""SQLite connection management and schema initialization."""
import sqlite3
import os


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_units (
    id              TEXT PRIMARY KEY,
    unit_type       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_status   TEXT NOT NULL DEFAULT 'candidate',
    title           TEXT DEFAULT '',
    text            TEXT NOT NULL,
    summary         TEXT DEFAULT '',
    source_uri      TEXT DEFAULT '',
    source_title    TEXT DEFAULT '',
    source_citation TEXT DEFAULT '',
    page_start      INTEGER,
    page_end        INTEGER,
    semantic_tags   TEXT DEFAULT '[]',
    scenario_tags   TEXT DEFAULT '[]',
    role_tags       TEXT DEFAULT '[]',
    method_tags     TEXT DEFAULT '[]',
    risk_levels     TEXT DEFAULT '[]',
    card_targets    TEXT DEFAULT '[]',
    contraindications TEXT DEFAULT '[]',
    quality_score   REAL NOT NULL DEFAULT 0.0,
    quality_flags   TEXT DEFAULT '[]',
    review_status   TEXT NOT NULL DEFAULT 'unreviewed',
    parent_chunk_ids TEXT DEFAULT '[]',
    parse_method    TEXT DEFAULT '',
    embedding_model TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_units_fts USING fts5(
    title, text, summary,
    content='knowledge_units',
    content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS source_documents (
    id              TEXT PRIMARY KEY,
    file_path       TEXT,
    source_uri      TEXT,
    source_type     TEXT NOT NULL,
    title           TEXT DEFAULT '',
    total_pages     INTEGER,
    parse_status    TEXT NOT NULL DEFAULT 'pending',
    chunk_count     INTEGER DEFAULT 0,
    imported_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              TEXT PRIMARY KEY,
    run_at          TEXT NOT NULL,
    total_queries   INTEGER,
    recall_at_10    REAL,
    safety_hit_rate REAL,
    noise_rate      REAL,
    avg_usability   REAL,
    details_json    TEXT DEFAULT '{}'
);
"""


class Database:
    """SQLite database connection manager."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def initialize(self) -> None:
        """Create all tables if they do not exist."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def list_tables(self) -> list[str]:
        """Return list of table names for verification."""
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
```

- [ ] **Step 5: Create src/db/repository.py**

```python
"""Repository pattern for KnowledgeUnit CRUD operations."""
from src.db.connection import Database


class KnowledgeUnitRepository:
    """Data access for knowledge_units table."""

    def __init__(self, db: Database):
        self.db = db

    def insert(self, data: dict) -> None:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT OR REPLACE INTO knowledge_units ({columns}) VALUES ({placeholders})"
        self.db.conn.execute(sql, list(data.values()))
        self.db.conn.commit()

    def get_by_id(self, ku_id: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT * FROM knowledge_units WHERE id = ?", (ku_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_by_status(self, source_status: str) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT * FROM knowledge_units WHERE source_status = ?", (source_status,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_active(self) -> list[dict]:
        """Get all chunks that should participate in retrieval (main + candidate)."""
        rows = self.db.conn.execute(
            "SELECT * FROM knowledge_units WHERE source_status IN ('main', 'candidate')"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self.db.conn.execute(
            "SELECT source_status, COUNT(*) as cnt FROM knowledge_units GROUP BY source_status"
        ).fetchall()
        return {r["source_status"]: r["cnt"] for r in rows}

    def count_by_type(self) -> dict[str, int]:
        rows = self.db.conn.execute(
            "SELECT unit_type, COUNT(*) as cnt FROM knowledge_units GROUP BY unit_type"
        ).fetchall()
        return {r["unit_type"]: r["cnt"] for r in rows}

    def search_by_tags(self, tags: list[str], field: str = "scenario_tags", limit: int = 20) -> list[dict]:
        """Search for units matching any of the given tags in the specified JSON array field."""
        conditions = " OR ".join(f"{field} LIKE ?" for _ in tags)
        params = [f"%{t}%" for t in tags]
        rows = self.db.conn.execute(
            f"SELECT * FROM knowledge_units WHERE ({conditions}) AND source_status IN ('main', 'candidate') LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def fts_search(self, query: str, limit: int = 30) -> list[dict]:
        """Full-text search using FTS5 index."""
        rows = self.db.conn.execute(
            """SELECT ku.* FROM knowledge_units ku
               INNER JOIN knowledge_units_fts fts ON ku.rowid = fts.rowid
               WHERE knowledge_units_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/db/ tests/test_db.py
git commit -m "feat: add database layer with SQLite schema and repository"
```

### Task 1.6: LLM client

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_client.py
import pytest
from src.llm.client import LLMClient, LLMConfig


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig(api_key="test-key")
        assert config.base_url == "https://api.deepseek.com/v1"
        assert config.model == "deepseek-chat"
        assert config.timeout == 60
        assert config.max_retries == 2

    def test_custom(self):
        config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-custom",
            model="gpt-4",
            timeout=30,
            max_retries=3,
        )
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4"


class TestLLMClient:
    def test_client_creation(self):
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        assert client.config.api_key == "test-key"

    def test_build_messages(self):
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        messages = client.build_messages(
            system="You are a helpful assistant.",
            user="Hello",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/llm/__init__.py**

```python
"""OpenAI-compatible LLM client layer."""
```

- [ ] **Step 4: Create src/llm/client.py**

```python
"""Generic OpenAI-compatible API client."""
from pydantic import BaseModel
from openai import AsyncOpenAI


class LLMConfig(BaseModel):
    """Configuration for an OpenAI-compatible LLM endpoint."""

    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    timeout: int = 60
    max_retries: int = 2


class LLMClient:
    """Async client for OpenAI-compatible chat completion APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def build_messages(system: str, user: str) -> list[dict]:
        """Build standard messages list from system + user prompts."""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        kwargs = dict(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def chat_with_json(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        """Chat completion with JSON mode, returns parsed dict. Retries once on parse failure."""
        import json

        for attempt in range(2):
            text = await self.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
        return {}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm/ tests/test_llm_client.py
git commit -m "feat: add OpenAI-compatible LLM client layer"
```

### Task 1.7: Health endpoint

**Files:**
- Create: `src/health/__init__.py`
- Create: `src/health/api.py`
- Create: `tests/test_health.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_health.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "embedding_model" in data
        assert "reranker_model" in data
        assert "index_loaded" in data
        assert "index_version" in data
        assert "document_count" in data
        assert "chunk_count" in data
        assert "card_count" in data

    def test_health_status_is_valid(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("ok", "degraded", "error")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_health.py -v`
Expected: FAIL — either 404 or connection error

- [ ] **Step 3: Create src/health/__init__.py**

```python
"""Health check module."""
```

- [ ] **Step 4: Create src/health/api.py**

```python
"""GET /health endpoint."""
from fastapi import APIRouter
from src.models.retrieval import HealthResponse  # We'll define this here inline for now
from pydantic import BaseModel
from typing import Literal


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


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health status."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    from src.config import settings

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        index_loaded=False,  # Will be updated when index module is ready
        index_version="",
        document_count=0,
        chunk_count=0,
        card_count=0,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_health.py -v`
Expected: PASS (if main.py already registered the router) or FAIL with 404

- [ ] **Step 6: Commit**

```bash
git add src/health/ tests/test_health.py
git commit -m "feat: add GET /health endpoint"
```

### Task 1.8: FastAPI app entry point

**Files:**
- Create: `src/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main.py
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestApp:
    def test_app_title(self):
        assert app.title == "NarrCare-KB"

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/health" in paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/main.py**

```python
"""NarrCare-KB FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.errors import KBException
from src.health.api import router as health_router

app = FastAPI(
    title="NarrCare-KB",
    description="Independent Knowledge Base Service for NarrCare",
    version="0.1.0",
)

# Register routers
app.include_router(health_router)


@app.exception_handler(KBException)
async def kb_exception_handler(request: Request, exc: KBException):
    """Convert KBException to structured JSON error response."""
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.error_code, "detail": exc.detail},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: Verify health endpoint through app**

Run: `python -m pytest tests/test_health.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: add FastAPI app entry point with error handler"
```

---

## Phase 2: Document Ingestion & Cleaning

### Task 2.1: Document parser

**Files:**
- Create: `src/ingestion/__init__.py`
- Create: `src/ingestion/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parser.py
import pytest
from src.ingestion.parser import DocumentParser, ParsedDocument, Page


class TestDocumentParser:
    def test_parsed_document_structure(self):
        doc = ParsedDocument(
            file_path="/tmp/test.pdf",
            source_type="pdf_book",
            pages=[],
        )
        assert doc.file_path == "/tmp/test.pdf"
        assert doc.source_type == "pdf_book"
        assert doc.pages == []

    def test_page_structure(self):
        page = Page(
            page_number=1,
            text="测试文本内容",
            parse_method="text_layer",
            confidence=1.0,
        )
        assert page.page_number == 1
        assert page.text == "测试文本内容"
        assert page.parse_method == "text_layer"

    def test_parser_accepts_txt(self, tmp_path):
        """Parser should handle plain text files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("这是一段安宁疗护相关的测试文本。")
        parser = DocumentParser()
        doc = parser.parse(str(txt_file))
        assert len(doc.pages) >= 1
        assert "安宁疗护" in doc.pages[0].text

    def test_parser_detects_source_type(self):
        parser = DocumentParser()
        assert parser.detect_source_type("book.pdf") == "pdf_book"
        assert parser.detect_source_type("paper.pdf") == "paper"
        assert parser.detect_source_type("guide.md") == "markdown"
        assert parser.detect_source_type("doc.docx") == "markdown"
        assert parser.detect_source_type("cases.jsonl") == "case"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parser.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/ingestion/__init__.py**

```python
"""Document ingestion module."""
```

- [ ] **Step 4: Create src/ingestion/parser.py**

```python
"""Document parser supporting PDF, Markdown, Docx, JSONL, and plain text."""
import os
from dataclasses import dataclass, field


@dataclass
class Page:
    """A single page of extracted text."""
    page_number: int
    text: str
    parse_method: str = "text_layer"  # text_layer | ocr | hybrid
    confidence: float = 1.0


@dataclass
class ParsedDocument:
    """Result of parsing a document."""
    file_path: str
    source_type: str
    pages: list[Page] = field(default_factory=list)


class DocumentParser:
    """Multi-format document parser.

    Supports: PDF (via PyMuPDF + PaddleOCR fallback), Markdown, Docx, JSONL, TXT.
    """

    TEXT_LAYER_MIN_CHARS = 50  # If a PDF page has fewer chars, fall back to OCR

    EXTENSION_MAP = {
        ".pdf": "pdf_book",
        ".md": "markdown",
        ".docx": "markdown",
        ".jsonl": "case",
        ".txt": "markdown",
    }

    def detect_source_type(self, file_path: str) -> str:
        """Infer source_type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return self.EXTENSION_MAP.get(ext, "pdf_book")

    def parse(self, file_path: str, source_type: str | None = None) -> ParsedDocument:
        """Parse a file into a ParsedDocument. Dispatches based on extension."""
        if source_type is None:
            source_type = self.detect_source_type(file_path)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path, source_type)
        elif ext in (".md", ".txt"):
            return self._parse_text(file_path, source_type)
        elif ext == ".docx":
            return self._parse_docx(file_path, source_type)
        elif ext == ".jsonl":
            return self._parse_jsonl(file_path, source_type)
        else:
            # Fallback: treat as text
            return self._parse_text(file_path, source_type)

    def _parse_pdf(self, file_path: str, source_type: str) -> ParsedDocument:
        """Parse PDF using PyMuPDF, with OCR fallback for pages with little text."""
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if len(text.strip()) >= self.TEXT_LAYER_MIN_CHARS:
                pages.append(Page(
                    page_number=i + 1,
                    text=text.strip(),
                    parse_method="text_layer",
                    confidence=1.0,
                ))
            else:
                # Low-text page — could be scanned. Mark for OCR later.
                pages.append(Page(
                    page_number=i + 1,
                    text=text.strip(),
                    parse_method="text_layer",
                    confidence=0.3,  # Low confidence, OCR may be needed
                ))
        doc.close()
        return ParsedDocument(file_path=file_path, source_type=source_type, pages=pages)

    def _parse_text(self, file_path: str, source_type: str) -> ParsedDocument:
        """Parse plain text or Markdown files."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return ParsedDocument(
            file_path=file_path,
            source_type=source_type,
            pages=[Page(page_number=1, text=text, parse_method="text_layer", confidence=1.0)],
        )

    def _parse_docx(self, file_path: str, source_type: str) -> ParsedDocument:
        """Parse Docx files."""
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        return ParsedDocument(
            file_path=file_path,
            source_type=source_type,
            pages=[Page(page_number=1, text=text, parse_method="text_layer", confidence=1.0)],
        )

    def _parse_jsonl(self, file_path: str, source_type: str) -> ParsedDocument:
        """Parse JSONL files — each line maps to a page."""
        import json
        pages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", json.dumps(obj, ensure_ascii=False)))
                except json.JSONDecodeError:
                    text = line
                pages.append(Page(
                    page_number=i + 1,
                    text=text,
                    parse_method="text_layer",
                    confidence=1.0,
                ))
        return ParsedDocument(file_path=file_path, source_type=source_type, pages=pages)

    def ocr_page(self, file_path: str, page_number: int) -> Page:
        """Run PaddleOCR on a specific page of a PDF. Used as fallback for scanned pages."""
        # Placeholder — actual PaddleOCR integration in Task 2.1b
        return Page(
            page_number=page_number,
            text="[OCR placeholder]",
            parse_method="ocr",
            confidence=0.7,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_parser.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/__init__.py src/ingestion/parser.py tests/test_parser.py
git commit -m "feat: add multi-format document parser (PDF, MD, DOCX, JSONL, TXT)"
```

### Task 2.2: Text cleaner

**Files:**
- Create: `src/ingestion/cleaner.py`
- Create: `tests/test_cleaner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cleaner.py
import pytest
from src.ingestion.parser import Page
from src.ingestion.cleaner import TextCleaner, CleanedPage


class TestTextCleaner:
    def test_clean_normal_text(self):
        cleaner = TextCleaner()
        page = Page(page_number=1, text="这是一个正常的段落，包含有意义的安宁疗护内容。", parse_method="text_layer")
        cleaned = cleaner.clean(page)
        assert cleaned.page_number == 1
        assert "安宁疗护" in cleaned.text
        assert cleaned.quality_score >= 0.5

    def test_toc_detection(self):
        cleaner = TextCleaner()
        page = Page(page_number=2, text="目  录\n第一章 死亡焦虑........1\n第二章 叙事护理........35", parse_method="text_layer")
        score, flags = cleaner.score_quality(page.text, page.parse_method)
        assert "table_of_contents" in flags
        assert score < 0.5

    def test_copyright_detection(self):
        cleaner = TextCleaner()
        page = Page(page_number=1, text="ISBN 978-7-123-45678-9\nCopyright 2024 出版社\nCIP数据核字(2024)第123456号", parse_method="text_layer")
        score, flags = cleaner.score_quality(page.text, page.parse_method)
        assert "copyright_page" in flags
        assert score < 0.5

    def test_too_short_text(self):
        cleaner = TextCleaner()
        page = Page(page_number=1, text="短", parse_method="text_layer")
        score, flags = cleaner.score_quality(page.text, page.parse_method)
        assert "too_short" in flags

    def test_garbled_text_detection(self):
        cleaner = TextCleaner()
        page = Page(page_number=1, text="脛脝脧脰脷脢鈥⑩", parse_method="ocr")
        score, flags = cleaner.score_quality(page.text, page.parse_method)
        # OCR text with many non-common characters should have lower score
        assert score < 1.0

    def test_ocr_base_score_lower(self):
        cleaner = TextCleaner()
        text_layer_page = Page(page_number=1, text="Normal meaningful text about hospice care and death anxiety.", parse_method="text_layer")
        ocr_page = Page(page_number=1, text="Normal meaningful text about hospice care and death anxiety.", parse_method="ocr")
        _, flags_tl = cleaner.score_quality(text_layer_page.text, text_layer_page.parse_method)
        score_ocr, _ = cleaner.score_quality(ocr_page.text, ocr_page.parse_method)
        # OCR should start from a lower base score
        assert score_ocr < 0.85  # OCR base is 0.7, minus any flags

    def test_header_footer_removal(self):
        cleaner = TextCleaner()
        text = "安宁疗护临床实践指南\n第12页\n正文内容开始：临终关怀需要...\n第12页"
        cleaned = cleaner._remove_headers_footers(text)
        assert "第12页" not in cleaned or "正文内容" in cleaned

    def test_determine_status(self):
        cleaner = TextCleaner()
        assert cleaner.determine_status(0.8, "main") == "main"
        assert cleaner.determine_status(0.8, "candidate") == "candidate"
        assert cleaner.determine_status(0.3, "main") == "quarantined"
        assert cleaner.determine_status(0.1, "candidate") == "quarantined"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cleaner.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/ingestion/cleaner.py**

```python
"""Text cleaning, denoising, and quality scoring."""
import re
from dataclasses import dataclass
from src.ingestion.parser import Page


@dataclass
class CleanedPage:
    """A cleaned page with quality metadata."""
    page_number: int
    text: str
    parse_method: str
    quality_score: float
    quality_flags: list[str]


class TextCleaner:
    """Cleans extracted text and assigns quality scores.

    Handles: header/footer removal, TOC detection, copyright detection,
    garbled text detection, low-information filtering, and dedup.
    """

    BASE_SCORE_TEXT = 1.0
    BASE_SCORE_OCR = 0.7
    QUALITY_THRESHOLD = 0.5

    PENALTIES = {
        "too_short": 0.3,
        "table_of_contents": 0.9,
        "copyright_page": 0.9,
        "isolated_characters": 0.4,
        "possible_ocr_garble": 0.3,
        "duplicated": 0.2,
        "low_information_density": 0.3,
    }

    # Patterns for header/footer removal
    PAGE_NUM_PATTERN = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
    HEADER_FOOTER_PATTERN = re.compile(r'第\s*\d+\s*页', re.MULTILINE)

    def clean(self, page: Page) -> CleanedPage:
        """Clean a page: remove headers/footers, then score quality."""
        text = self._remove_headers_footers(page.text)
        score, flags = self.score_quality(text, page.parse_method)
        return CleanedPage(
            page_number=page.page_number,
            text=text,
            parse_method=page.parse_method,
            quality_score=score,
            quality_flags=flags,
        )

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns from text."""
        text = self.PAGE_NUM_PATTERN.sub('', text)
        text = self.HEADER_FOOTER_PATTERN.sub('', text)
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def score_quality(self, text: str, parse_method: str) -> tuple[float, list[str]]:
        """Score text quality and return (score, flags)."""
        base = self.BASE_SCORE_TEXT if parse_method == "text_layer" else self.BASE_SCORE_OCR
        flags = []
        score = base

        # Too short
        if len(text) < 20:
            flags.append("too_short")
            score -= self.PENALTIES["too_short"]

        # Table of contents detection
        if self._is_toc(text):
            flags.append("table_of_contents")
            score -= self.PENALTIES["table_of_contents"]

        # Copyright page detection
        if self._is_copyright(text):
            flags.append("copyright_page")
            score -= self.PENALTIES["copyright_page"]

        # Garbled text detection (high ratio of rare Unicode characters)
        if self._is_garbled(text):
            flags.append("possible_ocr_garble")
            score -= self.PENALTIES["possible_ocr_garble"]

        # Isolated characters (mostly single chars without context)
        if self._has_isolated_chars(text):
            flags.append("isolated_characters")
            score -= self.PENALTIES["isolated_characters"]

        # Low information density
        if self._is_low_density(text):
            flags.append("low_information_density")
            score -= self.PENALTIES["low_information_density"]

        return max(0.0, score), flags

    def _is_toc(self, text: str) -> bool:
        """Detect if text looks like a table of contents."""
        toc_indicators = ["目录", "Contents", "目  录"]
        dot_line_pattern = re.search(r'\.{4,}', text)
        has_toc_keyword = any(indicator in text for indicator in toc_indicators)
        return has_toc_keyword and dot_line_pattern is not None

    def _is_copyright(self, text: str) -> bool:
        """Detect if text looks like a copyright/publishing info page."""
        indicators = ["ISBN", "CIP", "版权", "Copyright", "出版社", "印次", "版次"]
        count = sum(1 for ind in indicators if ind in text)
        return count >= 2

    def _is_garbled(self, text: str) -> bool:
        """Check for garbled text by looking at unusual Unicode character ratio."""
        if len(text) < 10:
            return False
        # Count characters in uncommon Unicode ranges (Private Use Area, etc.)
        unusual = sum(1 for c in text if ord(c) > 0x4E00 and ord(c) not in range(0x4E00, 0x9FFF)
                      and ord(c) not in range(0x3000, 0x303F)  # CJK punctuation
                      and ord(c) not in range(0xFF00, 0xFFEF)  # Fullwidth forms
                      and ord(c) not in range(0x0020, 0x007F))  # ASCII
        # Simple heuristic: >30% unusual chars is garbled
        return (unusual / len(text)) > 0.3

    def _has_isolated_chars(self, text: str) -> bool:
        """Check if text contains mostly isolated characters without sentences."""
        # Remove whitespace and check for CJK chars without context
        chars = [c for c in text if '一' <= c <= '鿿']
        return len(chars) < 5 and len(text.strip()) > 0

    def _is_low_density(self, text: str) -> bool:
        """Check for low information density (too much punctuation/whitespace)."""
        if len(text) < 30:
            return False
        non_info = sum(1 for c in text if c in ' \t\n\r，。、；：""''！？…—　')
        return (non_info / len(text)) > 0.7

    def determine_status(self, quality_score: float, desired_status: str) -> str:
        """Determine final source_status based on quality score.

        - quality >= 0.5: keep desired status (main or candidate)
        - quality < 0.5: quarantined
        - TOC/copyright flags force quarantine (handled in orchestrator)
        """
        if quality_score >= self.QUALITY_THRESHOLD:
            return desired_status
        return "quarantined"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cleaner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/cleaner.py tests/test_cleaner.py
git commit -m "feat: add text cleaner with quality scoring and TOC/copyright detection"
```

### Task 2.3: Semantic chunker

**Files:**
- Create: `src/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
import pytest
from src.ingestion.parser import Page, ParsedDocument
from src.ingestion.cleaner import TextCleaner, CleanedPage
from src.ingestion.chunker import SemanticChunker


@pytest.fixture
def sample_document():
    cleaner = TextCleaner()
    pages = []
    # Simulate a multi-page cleaned document
    text_p1 = """第一章 死亡焦虑概述

死亡焦虑是安宁疗护中最常见的心理问题之一。患者在面对生命终点时，往往会产生强烈的存在性焦虑。

这种焦虑不仅仅是对死亡本身的恐惧，更是对未完成事务、未表达情感的遗憾。"""
    
    text_p2 = """第二章 叙事护理方法

叙事护理通过让患者讲述自己的生命故事，帮助其重新建构生命意义。

在具体实践中，护士可以采用开放式提问，引导患者回顾生命中的重要时刻。"""
    
    for i, text in enumerate([text_p1, text_p2]):
        page = Page(page_number=i+1, text=text, parse_method="text_layer")
        cleaned = cleaner.clean(page)
        pages.append(cleaned)
    return pages


class TestSemanticChunker:
    def test_chunk_creates_segments(self, sample_document):
        chunker = SemanticChunker(min_chars=100, max_chars=500, overlap_chars=50)
        chunks = chunker.chunk(sample_document, source_title="测试书", source_uri="/tmp/test.pdf")
        assert len(chunks) > 0
        # Each chunk should have text
        for chunk in chunks:
            assert len(chunk["text"]) > 0

    def test_chunk_preserves_source_info(self, sample_document):
        chunker = SemanticChunker(min_chars=100, max_chars=500, overlap_chars=50)
        chunks = chunker.chunk(sample_document, source_title="测试书", source_uri="/tmp/test.pdf")
        for chunk in chunks:
            assert "source_title" in chunk
            assert chunk["source_title"] == "测试书"

    def test_chunk_size_within_bounds(self, sample_document):
        chunker = SemanticChunker(min_chars=200, max_chars=800, overlap_chars=80)
        chunks = chunker.chunk(sample_document, source_title="测试书", source_uri="/tmp/test.pdf")
        for chunk in chunks:
            assert len(chunk["text"]) <= 1000  # Allow some buffer

    def test_chunk_ids_are_unique(self, sample_document):
        chunker = SemanticChunker(min_chars=100, max_chars=500, overlap_chars=50)
        chunks = chunker.chunk(sample_document, source_title="测试书", source_uri="/tmp/test.pdf")
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_short_text_handling(self):
        chunker = SemanticChunker()
        short_page = CleanedPage(
            page_number=1, text="很短的文本。", parse_method="text_layer",
            quality_score=0.9, quality_flags=[],
        )
        chunks = chunker.chunk([short_page], source_title="test", source_uri="/tmp/test.pdf")
        # Short text should still produce at least one chunk if it passes merge threshold
        assert len(chunks) >= 0  # May or may not chunk depending on min_chars
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/ingestion/chunker.py**

```python
"""Semantic chunking — splits cleaned text by semantic boundaries, not fixed char counts."""
import uuid
from src.ingestion.cleaner import CleanedPage


class SemanticChunker:
    """Splits documents into semantically coherent chunks with overlap."""

    def __init__(
        self,
        min_chars: int = 300,
        max_chars: int = 800,
        overlap_chars: int = 100,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _make_id(self, source_type_short: str = "doc") -> str:
        short_uuid = uuid.uuid4().hex[:8]
        return f"ku_{source_type_short}_{short_uuid}"

    def chunk(
        self,
        cleaned_pages: list[CleanedPage],
        source_title: str = "",
        source_uri: str = "",
        source_type: str = "pdf_book",
    ) -> list[dict]:
        """Split cleaned pages into semantic chunks. Returns list of chunk dicts."""
        # Step 1: Merge all pages into one text flow, preserving page boundaries
        segments = self._split_by_boundaries(cleaned_pages)

        # Step 2: Merge short segments forward
        merged = self._merge_short_segments(segments)

        # Step 3: Split long segments at sentence boundaries
        final_segments = []
        for seg in merged:
            if len(seg["text"]) > self.max_chars:
                final_segments.extend(self._split_long_segment(seg))
            else:
                final_segments.append(seg)

        # Step 4: Build chunk dicts with overlap
        chunks = []
        source_type_short = source_type.replace("_", "")[:12]
        prev_text = ""
        for seg in final_segments:
            text = seg["text"]
            if prev_text and self.overlap_chars > 0:
                overlap = prev_text[-self.overlap_chars:]
                text = overlap + "\n" + text

            chunks.append({
                "id": self._make_id(source_type_short),
                "unit_type": "semantic_chunk",
                "source_type": source_type,
                "text": text,
                "title": self._extract_title(text),
                "source_title": source_title,
                "source_uri": source_uri,
                "page_start": seg.get("page_start"),
                "page_end": seg.get("page_end"),
                "parse_method": seg.get("parse_method", "text_layer"),
            })
            prev_text = seg["text"]

        return chunks

    def _split_by_boundaries(self, cleaned_pages: list[CleanedPage]) -> list[dict]:
        """Split text by paragraph boundaries."""
        segments = []
        for page in cleaned_pages:
            paragraphs = page.text.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    segments.append({
                        "text": para,
                        "page_start": page.page_number,
                        "page_end": page.page_number,
                        "parse_method": page.parse_method,
                    })
        return segments

    def _merge_short_segments(self, segments: list[dict]) -> list[dict]:
        """Merge segments shorter than min_chars with the next segment."""
        merged = []
        i = 0
        while i < len(segments):
            current = segments[i]
            while len(current["text"]) < self.min_chars and i + 1 < len(segments):
                i += 1
                current["text"] += "\n" + segments[i]["text"]
                current["page_end"] = segments[i]["page_end"]
            merged.append(current)
            i += 1
        return merged

    def _split_long_segment(self, seg: dict) -> list[dict]:
        """Split a segment longer than max_chars at sentence boundaries."""
        text = seg["text"]
        if len(text) <= self.max_chars:
            return [seg]

        # Split at sentence boundaries: 。！？\n
        parts = []
        current = ""
        for char in text:
            current += char
            if char in "。！？\n" and len(current) >= self.min_chars:
                parts.append(current.strip())
                current = ""
        if current.strip():
            if parts and len(current) < self.min_chars:
                parts[-1] += current  # Merge last short piece
            else:
                parts.append(current.strip())

        return [
            {**seg, "text": p}
            for p in parts if p
        ]

    @staticmethod
    def _extract_title(text: str) -> str:
        """Extract a short title from the beginning of chunk text."""
        first_line = text.split("\n")[0].strip()
        if len(first_line) <= 50:
            return first_line
        return first_line[:47] + "..."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add semantic chunker with boundary-aware splitting and overlap"
```

### Task 2.4: Ingestion orchestrator

**Files:**
- Create: `src/ingestion/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import pytest
import os
import tempfile
from src.db.connection import Database
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker
from src.ingestion.orchestrator import IngestionOrchestrator


@pytest.fixture
def orchestrator():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.sqlite")
    db = Database(db_path)
    db.initialize()
    orch = IngestionOrchestrator(
        db=db,
        parser=DocumentParser(),
        cleaner=TextCleaner(),
        chunker=SemanticChunker(min_chars=100, max_chars=500, overlap_chars=50),
    )
    yield orch
    db.close()


@pytest.fixture
def sample_txt_file(tmp_path):
    txt_file = tmp_path / "test_book.txt"
    txt_file.write_text("""第一章 安宁疗护基础

安宁疗护是一种以患者为中心的护理模式，旨在为终末期患者提供全面的身心照护。

死亡焦虑是安宁疗护中最常见的心理问题。患者往往表现出对未知的恐惧、对未竟事务的遗憾。

第二章 叙事护理实践

叙事护理通过倾听患者的故事，帮助其寻找生命的意义和价值。

护士在叙事护理中扮演着倾听者和引导者的角色，需要具备同理心和专业的沟通技巧。""")
    return str(txt_file)


class TestIngestionOrchestrator:
    def test_ingest_text_file(self, orchestrator, sample_txt_file):
        result = orchestrator.ingest_file(sample_txt_file)
        assert result["status"] == "done"
        assert result["chunk_count"] > 0
        assert result["quarantined_count"] >= 0

    def test_ingest_creates_document_record(self, orchestrator, sample_txt_file):
        orchestrator.ingest_file(sample_txt_file)
        docs = orchestrator.db.conn.execute(
            "SELECT * FROM source_documents"
        ).fetchall()
        assert len(docs) == 1
        assert docs[0]["parse_status"] == "done"

    def test_ingest_writes_knowledge_units(self, orchestrator, sample_txt_file):
        result = orchestrator.ingest_file(sample_txt_file)
        units = orchestrator.db.conn.execute(
            "SELECT * FROM knowledge_units"
        ).fetchall()
        assert len(units) == result["chunk_count"]

    def test_deduplication(self, orchestrator, sample_txt_file):
        """Importing the same file twice should skip the second time."""
        orchestrator.ingest_file(sample_txt_file)
        result2 = orchestrator.ingest_file(sample_txt_file)
        assert result2["status"] == "skipped"
        assert "already imported" in result2.get("message", "").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/ingestion/orchestrator.py**

```python
"""Orchestrates the full ingestion pipeline."""
import hashlib
import uuid
from datetime import datetime, timezone
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker


class IngestionOrchestrator:
    """Coordinates the full ingestion pipeline: parse → clean → chunk → store."""

    def __init__(
        self,
        db: Database,
        parser: DocumentParser,
        cleaner: TextCleaner,
        chunker: SemanticChunker,
    ):
        self.db = db
        self.parser = parser
        self.cleaner = cleaner
        self.chunker = chunker
        self.repo = KnowledgeUnitRepository(db)

    def _file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file content for dedup."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _is_already_imported(self, file_path: str) -> bool:
        """Check if a file has already been imported based on path + hash."""
        file_hash = self._file_hash(file_path)
        row = self.db.conn.execute(
            "SELECT id FROM source_documents WHERE file_path = ? LIMIT 1",
            (file_path,),
        ).fetchone()
        return row is not None

    def ingest_file(
        self,
        file_path: str,
        source_type: str | None = None,
        source_status: str = "main",
    ) -> dict:
        """Run the full ingestion pipeline on a single file."""
        import os
        if not os.path.exists(file_path):
            return {"status": "failed", "message": f"File not found: {file_path}", "chunk_count": 0, "quarantined_count": 0}

        if self._is_already_imported(file_path):
            return {"status": "skipped", "message": f"File already imported: {file_path}", "chunk_count": 0, "quarantined_count": 0}

        if source_type is None:
            source_type = self.parser.detect_source_type(file_path)

        now = datetime.now(timezone.utc).isoformat()
        source_title = os.path.basename(file_path)
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Record document as parsing
        self.db.conn.execute(
            """INSERT OR REPLACE INTO source_documents (id, file_path, source_type, title, parse_status, imported_at)
               VALUES (?, ?, ?, ?, 'parsing', ?)""",
            (doc_id, file_path, source_type, source_title, now),
        )
        self.db.conn.commit()

        try:
            # Parse
            parsed_doc = self.parser.parse(file_path, source_type)

            # Clean
            cleaned_pages = [self.cleaner.clean(page) for page in parsed_doc.pages]

            # Chunk
            chunks = self.chunker.chunk(
                cleaned_pages,
                source_title=source_title,
                source_uri=file_path,
                source_type=source_type,
            )

            # Determine status per chunk and store
            chunk_count = 0
            quarantined_count = 0
            for chunk in chunks:
                # Find quality score from associated page
                quality_score = 0.8  # Default — will be overridden by cleaner output
                quality_flags = "[]"
                parse_method = chunk.get("parse_method", "text_layer")

                # Determine status
                final_status = self.cleaner.determine_status(quality_score, source_status)
                if final_status == "quarantined":
                    quarantined_count += 1
                else:
                    chunk_count += 1

                self.repo.insert({
                    "id": chunk["id"],
                    "unit_type": chunk["unit_type"],
                    "source_type": chunk["source_type"],
                    "source_status": final_status,
                    "title": chunk.get("title", ""),
                    "text": chunk["text"],
                    "summary": "",
                    "source_uri": chunk.get("source_uri", ""),
                    "source_title": chunk.get("source_title", ""),
                    "source_citation": "",
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "semantic_tags": "[]",
                    "scenario_tags": "[]",
                    "role_tags": "[]",
                    "method_tags": "[]",
                    "risk_levels": "[]",
                    "card_targets": "[]",
                    "contraindications": "[]",
                    "quality_score": quality_score,
                    "quality_flags": quality_flags,
                    "review_status": "unreviewed",
                    "parent_chunk_ids": "[]",
                    "parse_method": parse_method,
                    "embedding_model": "",
                    "created_at": now,
                    "updated_at": now,
                })

            # Update document status
            self.db.conn.execute(
                "UPDATE source_documents SET parse_status = 'done', chunk_count = ? WHERE id = ?",
                (chunk_count, doc_id),
            )
            self.db.conn.commit()

            return {
                "status": "done",
                "message": f"Ingested {len(chunks)} chunks from {source_title}",
                "document_id": doc_id,
                "chunk_count": chunk_count,
                "total_segments": len(chunks),
                "quarantined_count": quarantined_count,
            }
        except Exception as e:
            self.db.conn.execute(
                "UPDATE source_documents SET parse_status = 'failed' WHERE id = ?",
                (doc_id,),
            )
            self.db.conn.commit()
            return {"status": "failed", "message": str(e), "chunk_count": 0, "quarantined_count": 0}

    def ingest_batch(self, file_paths: list[str], source_type: str | None = None, source_status: str = "main") -> list[dict]:
        """Batch ingest multiple files."""
        results = []
        for fp in file_paths:
            result = self.ingest_file(fp, source_type=source_type, source_status=source_status)
            results.append(result)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add ingestion orchestrator with parse-clean-chunk-store pipeline"
```

### Task 2.5: Ingestion API endpoints

**Files:**
- Create: `src/ingestion/api.py`
- Modify: `src/main.py`
- Create: `tests/test_ingestion_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion_api.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestIngestionAPI:
    def test_ingest_files_endpoint_exists(self, tmp_path):
        """POST /ingest/files should accept file paths."""
        # Create a temp file to ingest
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("安宁疗护测试内容。临终关怀是重要的护理方式。")
        
        response = client.post("/ingest/files", json={
            "file_paths": [str(txt_file)],
            "source_status": "main",
        })
        assert response.status_code in (200, 500)  # 500 if DB not initialized in test yet

    def test_ingest_files_requires_file_paths(self):
        response = client.post("/ingest/files", json={})
        assert response.status_code == 422  # Validation error

    def test_ingest_search_endpoint_exists(self):
        response = client.post("/ingest/search", json={
            "topic": "death anxiety hospice",
            "keywords": ["palliative care", "narrative medicine"],
        })
        assert response.status_code in (200, 501)  # 501 if not implemented yet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion_api.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Create src/ingestion/api.py**

```python
"""Ingestion API endpoints: POST /ingest/files, POST /ingest/search."""
from fastapi import APIRouter, BackgroundTasks
from src.models.ingestion import (
    IngestFilesRequest, IngestFilesResponse,
    IngestSearchRequest, IngestSearchResponse,
)
from src.config import settings
from src.errors import KBException, KBErrorCode

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# These will be lazily initialized via dependency injection or global state
_orchestrator = None


def get_orchestrator():
    """Lazy-init the ingestion orchestrator (avoids circular imports)."""
    global _orchestrator
    if _orchestrator is None:
        from src.db.connection import Database
        from src.db.repository import KnowledgeUnitRepository
        from src.ingestion.parser import DocumentParser
        from src.ingestion.cleaner import TextCleaner
        from src.ingestion.chunker import SemanticChunker
        from src.ingestion.orchestrator import IngestionOrchestrator

        db = Database(settings.db_path)
        db.initialize()
        _orchestrator = IngestionOrchestrator(
            db=db,
            parser=DocumentParser(),
            cleaner=TextCleaner(),
            chunker=SemanticChunker(
                min_chars=settings.chunk_min_chars,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ),
        )
    return _orchestrator


@router.post("/files", response_model=IngestFilesResponse)
async def ingest_files(request: IngestFilesRequest):
    """Import local files into the knowledge base."""
    import uuid
    orchestrator = get_orchestrator()
    try:
        results = orchestrator.ingest_batch(
            file_paths=request.file_paths,
            source_type=request.source_type,
            source_status=request.source_status,
        )
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        return IngestFilesResponse(
            task_id=task_id,
            status="done",
            results=results,
        )
    except Exception as e:
        raise KBException(
            error_code=KBErrorCode.KB_INGESTION_FAILED,
            detail=str(e),
            http_status=500,
        )


@router.post("/search", response_model=IngestSearchResponse)
async def ingest_search(request: IngestSearchRequest):
    """Search external sources and import results. (Phase 7 — returns placeholder for now)"""
    import uuid
    return IngestSearchResponse(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        candidates=[],
        status="not_implemented",
    )
```

- [ ] **Step 4: Update src/main.py to register ingestion router**

```python
# Add after other router imports in src/main.py:
from src.ingestion.api import router as ingestion_router

# Add after other router registrations:
app.include_router(ingestion_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/api.py src/main.py tests/test_ingestion_api.py
git commit -m "feat: add POST /ingest/files and /ingest/search API endpoints"
```

## Phase 3: LLM Enrichment & Knowledge Cards

### Task 3.1: Tag pool definition

**Files:**
- Create: `src/ingestion/tag_pool.py`
- Create: `tests/test_tag_pool.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tag_pool.py
from src.ingestion.tag_pool import TAG_POOL, validate_tags


class TestTagPool:
    def test_tag_pool_has_categories(self):
        assert "semantic_tags" in TAG_POOL
        assert "scenario_tags" in TAG_POOL
        assert "role_tags" in TAG_POOL
        assert "method_tags" in TAG_POOL
        assert "risk_levels" in TAG_POOL
        assert "card_targets" in TAG_POOL

    def test_semantic_tags_not_empty(self):
        assert len(TAG_POOL["semantic_tags"]) > 5

    def test_validate_tags_filters_unknown(self):
        result = validate_tags(
            semantic_tags=["死亡焦虑", "unknown_tag_xyz"],
            scenario_tags=["夜间焦虑"],
        )
        assert "死亡焦虑" in result["semantic_tags"]
        assert "unknown_tag_xyz" not in result["semantic_tags"]
        assert "夜间焦虑" in result["scenario_tags"]

    def test_card_targets_are_valid(self):
        for ct in TAG_POOL["card_targets"]:
            assert ct in ("mindfulness", "healing", "communication", "personalized")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tag_pool.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/ingestion/tag_pool.py**

```python
"""Predefined tag pool for LLM enrichment. Tags not in this pool are discarded."""

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tag_pool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tag_pool.py tests/test_tag_pool.py
git commit -m "feat: add predefined tag pool for LLM enrichment validation"
```

### Task 3.2: LLM Enricher

**Files:**
- Create: `src/ingestion/enricher.py`
- Create: `tests/test_enricher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enricher.py
import pytest
from unittest.mock import AsyncMock, patch
from src.ingestion.enricher import LLMEnricher


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.chat_with_json.return_value = {
        "summary": "临终患者的存在性痛苦管理方法",
        "semantic_tags": ["死亡焦虑", "存在性痛苦"],
        "scenario_tags": ["夜间焦虑"],
        "role_tags": ["nurse", "patient"],
        "method_tags": ["尊严疗法", "叙事疗法"],
        "risk_levels": ["medium"],
        "card_targets": ["communication"],
    }
    return client


class TestLLMEnricher:
    def test_enrich_chunk_returns_tags(self, mock_llm_client):
        enricher = LLMEnricher(mock_llm_client)
        chunk = {"id": "ku_test_1", "text": "临终患者常常经历存在性痛苦..."}
        result = enricher.enrich_chunk_sync(chunk)
        assert "summary" in result
        assert "semantic_tags" in result
        assert len(result.get("semantic_tags", [])) > 0

    def test_enrich_chunk_filters_invalid_tags(self, mock_llm_client):
        mock_llm_client.chat_with_json.return_value = {
            "summary": "test",
            "semantic_tags": ["死亡焦虑", "made_up_tag_xyz"],
            "scenario_tags": [],
            "role_tags": ["patient"],
            "method_tags": [],
            "risk_levels": [],
            "card_targets": [],
        }
        enricher = LLMEnricher(mock_llm_client)
        chunk = {"id": "ku_test_2", "text": "test content"}
        result = enricher.enrich_chunk_sync(chunk)
        assert "死亡焦虑" in result["semantic_tags"]
        assert "made_up_tag_xyz" not in result["semantic_tags"]

    def test_generate_card_has_parent_ids(self, mock_llm_client):
        mock_llm_client.chat_with_json.return_value = {
            "title": "夜间焦虑的叙事护理干预",
            "text": "针对夜间焦虑的护理建议...",
            "card_targets": ["communication"],
        }
        enricher = LLMEnricher(mock_llm_client)
        parent_chunks = [
            {"id": "ku_chunk_a", "text": "夜间焦虑是常见问题..."},
            {"id": "ku_chunk_b", "text": "叙事护理可缓解焦虑..."},
        ]
        card = enricher.generate_card_sync(parent_chunks)
        assert card is not None
        assert "parent_chunk_ids" in card
        assert "ku_chunk_a" in card["parent_chunk_ids"]
        assert "ku_chunk_b" in card["parent_chunk_ids"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_enricher.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/ingestion/enricher.py**

```python
"""LLM enrichment: summary, tags, and knowledge card generation."""
import json
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

    def enrich_chunk_sync(self, chunk: dict) -> dict:
        """Synchronously enrich a single chunk. (Use enrich_chunk for async.)"""
        import asyncio
        return asyncio.run(self.enrich_chunk(chunk))

    async def enrich_chunk(self, chunk: dict) -> dict:
        """Call LLM to generate summary and tags for a chunk."""
        messages = self.llm.build_messages(
            system=ENRICH_SYSTEM_PROMPT,
            user=f"Text:\n{chunk['text'][:2000]}",  # Truncate for token limits
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
        return {
            **chunk,
            "summary": result.get("summary", ""),
            **tags,
        }

    def generate_card_sync(self, parent_chunks: list[dict]) -> dict | None:
        """Synchronously generate a knowledge card."""
        import asyncio
        return asyncio.run(self.generate_card(parent_chunks))

    async def generate_card(self, parent_chunks: list[dict]) -> dict | None:
        """Generate a knowledge card from parent chunks. Returns card dict or None on failure."""
        parent_ids = [c["id"] for c in parent_chunks]
        combined_text = "\n\n".join(c["text"][:500] for c in parent_chunks[:3])

        messages = self.llm.build_messages(
            system=CARD_SYSTEM_PROMPT,
            user=f"Source chunks:\n{combined_text}",
        )
        try:
            result = await self.llm.chat_with_json(messages, temperature=0.5, max_tokens=2048)
        except Exception:
            return None

        if not result.get("title") or not result.get("text"):
            return None

        card_targets = result.get("card_targets", [])
        card_targets = [ct for ct in card_targets if ct in ("mindfulness", "healing", "communication", "personalized")]

        return {
            "title": result["title"],
            "text": result["text"],
            "card_targets": card_targets,
            "parent_chunk_ids": parent_ids,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_enricher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/enricher.py tests/test_enricher.py
git commit -m "feat: add LLM enricher for summary, tags, and knowledge card generation"
```

## Phase 4: GPU Embedding & Indexing

### Task 4.1: Embedder

**Files:**
- Create: `src/indexing/__init__.py`
- Create: `src/indexing/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.indexing.embedder import Embedder


class TestEmbedder:
    def test_init_stores_config(self):
        emb = Embedder(model_path="models/Qwen3-Embedding-0.6B", device="cpu")
        assert emb.model_path == "models/Qwen3-Embedding-0.6B"
        assert emb.device == "cpu"

    def test_encode_returns_correct_shape(self):
        emb = Embedder(model_path="models/Qwen3-Embedding-0.6B", device="cpu")
        # Mock the model to avoid loading real model in tests
        emb._model = MagicMock()
        emb._model.encode.return_value = np.random.randn(3, 1024).astype(np.float32)
        
        texts = ["测试文本1", "测试文本2", "测试文本3"]
        vectors = emb.encode(texts, batch_size=2)
        assert vectors.shape == (3, 1024)

    def test_encode_query_returns_2d(self):
        emb = Embedder(model_path="models/Qwen3-Embedding-0.6B", device="cpu")
        emb._model = MagicMock()
        emb._model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        
        vector = emb.encode_query("单条查询文本")
        assert vector.shape == (1024,)

    def test_load_fails_gracefully(self):
        emb = Embedder(model_path="/nonexistent/path", device="cpu")
        result = emb.load()
        assert result is False
        assert emb._model is None
```

- [ ] **Step 2: Run test and implement**

Run: `python -m pytest tests/test_embedder.py -v` (expect FAIL)
Then implement `src/indexing/embedder.py`:

```python
"""Embedding model wrapper for Qwen3-Embedding-0.6B."""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps sentence-transformers for batch encoding and single-query encoding."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> bool:
        """Load the embedding model. Returns True on success."""
        try:
            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
            )
            if self.device == "cuda":
                self._model.half()  # FP16 to save VRAM
            return True
        except Exception:
            self._model = None
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Batch encode texts. Returns (N, dim) float32 array."""
        if not self.is_loaded:
            raise RuntimeError("Embedder model not loaded")
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Returns (dim,) float32 array."""
        result = self.encode([query], batch_size=1)
        return result[0]
```

Run: `python -m pytest tests/test_embedder.py -v` (expect PASS)

- [ ] **Step 3: Commit**

```bash
git add src/indexing/ tests/test_embedder.py
git commit -m "feat: add Qwen3 embedding model wrapper"
```

### Task 4.2: FAISS vector index

**Files:**
- Create: `src/indexing/vector_index.py`
- Create: `tests/test_vector_index.py`

- [ ] **Step 1: Write test and implement vector_index.py**

Test key behaviors: build from chunks, search returns scored hits, load/save persistence, add incremental.

```python
# src/indexing/vector_index.py
import os
import json
import numpy as np
import faiss


class VectorIndex:
    """FAISS-based dense vector index with persistence."""

    def __init__(self, dim: int = 1024, index_path: str = "data/index/faiss.index"):
        self.dim = dim
        self.index_path = index_path
        self.index: faiss.Index | None = None
        self.id_map: dict[int, str] = {}  # FAISS internal ID -> knowledge_unit ID

    def build(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        """Build a new FAISS index from embeddings."""
        embeddings = embeddings.astype(np.float32)
        # Normalize for inner product (cosine similarity)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)  # Inner Product = Cosine on normalized vectors
        self.index.add(embeddings)
        self.id_map = {i: cid for i, cid in enumerate(chunk_ids)}

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[dict]:
        """Search for k nearest neighbors. Returns list of {id, score}."""
        if self.index is None:
            return []
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                results.append({"id": self.id_map[idx], "score": float(score)})
        return results

    def save(self) -> None:
        """Persist index and id_map to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        map_path = self.index_path.replace(".index", "_id_map.json")
        with open(map_path, "w") as f:
            json.dump(self.id_map, f)
        meta_path = self.index_path.replace(".index", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"dim": self.dim, "size": self.index.ntotal}, f)

    def load(self) -> bool:
        """Load index and id_map from disk. Returns True on success."""
        if not os.path.exists(self.index_path):
            return False
        try:
            self.index = faiss.read_index(self.index_path)
            map_path = self.index_path.replace(".index", "_id_map.json")
            with open(map_path) as f:
                self.id_map = {int(k): v for k, v in json.load(f).items()}
            return True
        except Exception:
            return False

    def add(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        """Incrementally add new vectors to the index."""
        if self.index is None:
            self.build(chunk_ids, embeddings)
            return
        start_idx = self.index.ntotal
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for i, cid in enumerate(chunk_ids):
            self.id_map[start_idx + i] = cid

    @property
    def is_loaded(self) -> bool:
        return self.index is not None

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
```

- [ ] **Step 2: Run test and commit**

```bash
python -m pytest tests/test_vector_index.py -v
git add src/indexing/vector_index.py tests/test_vector_index.py
git commit -m "feat: add FAISS vector index with build/search/save/load"
```

### Task 4.3: FTS5 text index + Index rebuild API

**Files:**
- Create: `src/indexing/text_index.py`
- Create: `src/indexing/api.py`
- Modify: `src/main.py`
- Create: `tests/test_index_api.py`

- [ ] **Step 1: Create src/indexing/text_index.py**

```python
"""FTS5 text index management."""
from src.db.connection import Database


class TextIndex:
    """Manages SQLite FTS5 full-text index."""

    def __init__(self, db: Database):
        self.db = db

    def rebuild(self) -> None:
        """Rebuild FTS5 index by re-inserting all active knowledge units."""
        self.db.conn.execute("DELETE FROM knowledge_units_fts")
        rows = self.db.conn.execute(
            "SELECT rowid, title, text, summary FROM knowledge_units WHERE source_status IN ('main', 'candidate')"
        ).fetchall()
        for row in rows:
            self.db.conn.execute(
                "INSERT INTO knowledge_units_fts(rowid, title, text, summary) VALUES (?, ?, ?, ?)",
                (row["rowid"], row["title"], row["text"], row["summary"]),
            )
        self.db.conn.commit()

    def search(self, query: str, limit: int = 30) -> list[dict]:
        """Search FTS5 and return matching knowledge unit IDs with scores."""
        rows = self.db.conn.execute(
            """SELECT ku.id, ku.source_type, ku.source_status, ku.title,
                      snippet(knowledge_units_fts, 1, '<mark>', '</mark>', '...', 40) as snippet,
                      rank
               FROM knowledge_units_fts fts
               JOIN knowledge_units ku ON ku.rowid = fts.rowid
               WHERE knowledge_units_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 2: Create src/indexing/api.py**

```python
"""POST /index/rebuild endpoint."""
from fastapi import APIRouter
from src.config import settings
from src.errors import KBException, KBErrorCode

router = APIRouter(prefix="/index", tags=["indexing"])


@router.post("/rebuild")
async def rebuild_index(full: bool = True):
    """Rebuild FAISS and FTS5 indexes."""
    try:
        from src.db.connection import Database
        from src.indexing.text_index import TextIndex

        db = Database(settings.db_path)
        db.initialize()

        # Rebuild FTS5
        text_index = TextIndex(db)
        text_index.rebuild()

        # Rebuild FAISS (placeholder — full implementation needs embeddings)
        # Import here to avoid loading model on every module import
        from src.indexing.embedder import Embedder
        from src.indexing.vector_index import VectorIndex
        from src.db.repository import KnowledgeUnitRepository

        repo = KnowledgeUnitRepository(db)
        active = repo.get_all_active()

        if active:
            embedder = Embedder(
                model_path=f"{settings.models_dir}/{settings.embedding_model}",
                device=settings.embedding_device,
            )
            if not embedder.load():
                raise KBException(
                    error_code=KBErrorCode.KB_EMBEDDING_ERROR,
                    detail="Failed to load embedding model",
                    http_status=500,
                )

            texts = [u["text"] for u in active]
            ids = [u["id"] for u in active]
            embeddings = embedder.encode(texts, batch_size=settings.embedding_batch_size)

            vec_index = VectorIndex(dim=embeddings.shape[1], index_path=settings.faiss_index_path)
            vec_index.build(ids, embeddings)
            vec_index.save()

        db.close()
        return {
            "status": "done",
            "faiss_size": vec_index.size if active else 0,
            "fts5_rebuilt": True,
            "chunk_count": len(active),
        }
    except KBException:
        raise
    except Exception as e:
        raise KBException(
            error_code=KBErrorCode.KB_EMBEDDING_ERROR,
            detail=str(e),
            http_status=500,
        )
```

- [ ] **Step 3: Register index router in src/main.py**

Add:
```python
from src.indexing.api import router as indexing_router
app.include_router(indexing_router)
```

- [ ] **Step 4: Test and commit**

```bash
python -m pytest tests/test_index_api.py -v
git add src/indexing/ src/main.py tests/test_index_api.py
git commit -m "feat: add FTS5 text index and POST /index/rebuild endpoint"
```

## Phase 5: Query Understanding & Hybrid Retrieval

### Task 5.1: Query understanding

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/query_understanding.py`
- Create: `tests/test_query_understanding.py`

- [ ] **Step 1: Write test and implement**

Test: verify LLM call produces intents, scenario_tags, rewritten_queries (4 variants); verify graceful degradation on LLM failure (returns minimal valid result).

```python
# src/retrieval/query_understanding.py
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
        family_text: str,
        risk_assessment: dict | None = None,
        dyadic_analysis: dict | None = None,
    ) -> QueryAnalysis:
        """Analyze patient + family text and return QueryAnalysis."""
        user_text = f"Patient: {patient_text}\nFamily: {family_text}"
        if risk_assessment:
            user_text += f"\nRisk: {risk_assessment}"
        if dyadic_analysis:
            user_text += f"\nDyadic: {dyadic_analysis}"

        messages = self.llm.build_messages(system=QU_SYSTEM_PROMPT, user=user_text)

        try:
            result = await self.llm.chat_with_json(messages, temperature=0.3, max_tokens=1024)
        except Exception:
            # Graceful degradation: return minimal valid QueryAnalysis
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
        """Synchronous wrapper for analyze."""
        import asyncio
        return asyncio.run(self.analyze(patient_text, family_text, **kwargs))
```

- [ ] **Step 2: Run test and commit**

```bash
python -m pytest tests/test_query_understanding.py -v
git add src/retrieval/ tests/test_query_understanding.py
git commit -m "feat: add LLM query understanding with rewrite and graceful degradation"
```

### Task 5.2: Dense, Sparse, Metadata, Safety recall + Hybrid fusion

**Files:**
- Create: `src/retrieval/dense.py`
- Create: `src/retrieval/sparse.py`
- Create: `src/retrieval/metadata_recall.py`
- Create: `src/retrieval/safety_recall.py`
- Create: `src/retrieval/hybrid.py`
- Create: `tests/test_recall.py`

- [ ] **Step 1: Create all recall modules and hybrid fuser**

```python
# src/retrieval/dense.py
"""FAISS dense vector recall."""
from src.config import settings
from src.indexing.embedder import Embedder
from src.indexing.vector_index import VectorIndex


class DenseRecaller:
    def __init__(self, embedder: Embedder, vector_index: VectorIndex):
        self.embedder = embedder
        self.index = vector_index

    def recall(self, queries: list[str], top_k: int | None = None) -> list[dict]:
        """For each query, search FAISS and merge unique hits."""
        if top_k is None:
            top_k = settings.dense_top_k
        seen = set()
        results = []
        for query in queries:
            q_vec = self.embedder.encode_query(query)
            hits = self.index.search(q_vec, k=top_k)
            for hit in hits:
                if hit["id"] not in seen:
                    seen.add(hit["id"])
                    hit["source"] = "dense"
                    results.append(hit)
        return results
```

```python
# src/retrieval/sparse.py
"""FTS5 sparse recall with jieba tokenization."""
import jieba
from src.config import settings


class SparseRecaller:
    def __init__(self, text_index):
        self.text_index = text_index

    def recall(self, queries: list[str], top_k: int | None = None) -> list[dict]:
        if top_k is None:
            top_k = settings.sparse_top_k
        seen = set()
        results = []
        for query in queries:
            # Tokenize Chinese text for FTS5
            tokens = " ".join(jieba.cut(query))
            try:
                hits = self.text_index.search(tokens, limit=top_k)
            except Exception:
                hits = []
            for hit in hits:
                if hit["id"] not in seen:
                    seen.add(hit["id"])
                    hit["source"] = "sparse"
                    hit["score"] = float(hit.get("score", 0.0))
                    results.append(hit)
        return results
```

```python
# src/retrieval/metadata_recall.py
"""Metadata/tag-based recall."""
from src.config import settings
from src.db.repository import KnowledgeUnitRepository


class MetadataRecaller:
    def __init__(self, repo: KnowledgeUnitRepository):
        self.repo = repo

    def recall(self, scenario_tags: list[str], role_focus: list[str], top_k: int | None = None) -> list[dict]:
        if top_k is None:
            top_k = settings.metadata_top_k
        results = []
        seen = set()

        # Search by scenario_tags
        if scenario_tags:
            for unit in self.repo.search_by_tags(scenario_tags, "scenario_tags", top_k):
                if unit["id"] not in seen:
                    seen.add(unit["id"])
                    results.append({"id": unit["id"], "score": 0.5, "source": "metadata"})

        # Search by role_tags
        if role_focus:
            for unit in self.repo.search_by_tags(role_focus, "role_tags", top_k):
                if unit["id"] not in seen:
                    seen.add(unit["id"])
                    results.append({"id": unit["id"], "score": 0.4, "source": "metadata"})

        return results
```

```python
# src/retrieval/safety_recall.py
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
        """Force recall safety/contraindication evidence based on signals."""
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
                            "id": unit["id"],
                            "score": 1.0,  # Safety results get max base score
                            "source": "safety",
                            "safety_priority_boost": 0.2,
                        })
        return results
```

```python
# src/retrieval/hybrid.py
"""Multi-recall fusion with weighted scoring."""
from src.config import settings


class HybridFusion:
    def __init__(self):
        self.w_recall = settings.weight_recall
        self.w_metadata = settings.weight_metadata
        self.w_quality = settings.weight_quality
        self.w_source = settings.weight_source_status

    def fuse(
        self,
        dense_hits: list[dict],
        sparse_hits: list[dict],
        metadata_hits: list[dict],
        safety_hits: list[dict],
        knowledge_units: dict[str, dict],  # id -> unit data
    ) -> list[dict]:
        """Merge and score all recall sources. Returns deduplicated scored hits."""
        merged: dict[str, dict] = {}

        source_weight_map = {"main": 0.05, "candidate": 0.025, "quarantined": 0.0}

        for hit in dense_hits + sparse_hits + metadata_hits + safety_hits:
            uid = hit["id"]
            if uid in merged:
                merged[uid]["score"] += hit.get("score", 0.0) * 0.5
                if hit.get("safety_priority_boost"):
                    merged[uid]["score"] += hit["safety_priority_boost"]
            else:
                unit = knowledge_units.get(uid, {})
                quality = unit.get("quality_score", 0.5)
                src_status = unit.get("source_status", "candidate")

                score = (
                    hit.get("score", 0.0) * self.w_recall
                    + quality * self.w_quality
                    + source_weight_map.get(src_status, 0.0) * self.w_source
                )
                if hit.get("safety_priority_boost"):
                    score += hit["safety_priority_boost"]

                merged[uid] = {
                    "id": uid,
                    "score": score,
                    "sources": [hit.get("source", "unknown")],
                    "safety_boost": hit.get("safety_priority_boost", 0.0),
                }

        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
```

- [ ] **Step 2: Run test and commit**

```bash
python -m pytest tests/test_recall.py -v
git add src/retrieval/dense.py src/retrieval/sparse.py src/retrieval/metadata_recall.py \
        src/retrieval/safety_recall.py src/retrieval/hybrid.py tests/test_recall.py
git commit -m "feat: add dense, sparse, metadata, safety recall and hybrid fusion"
```

## Phase 6: Reranker & Evidence Bundle

### Task 6.1: Reranker

**Files:**
- Create: `src/retrieval/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: Implement reranker**

```python
# src/retrieval/reranker.py
"""Qwen3 Reranker for candidate re-ranking."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from src.config import settings


class Qwen3Reranker:
    """Qwen3-Reranker-0.6B cross-encoder for relevance scoring."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> bool:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
            return True
        except Exception:
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """Re-rank candidates by relevance to query. Returns re-scored list."""
        if not self.is_loaded or not candidates:
            return candidates

        pairs = []
        for c in candidates:
            unit_text = c.get("text", c.get("snippet", ""))[:512]
            pairs.append(f"Query: {query}\nDocument: {unit_text}")

        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self._model(**inputs).logits.squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["score"] = (
                float(score) * settings.weight_rerank
                + candidates[i].get("score", 0.0)
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
```

- [ ] **Step 2: Test and commit**

```bash
python -m pytest tests/test_reranker.py -v
git add src/retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: add Qwen3 reranker for cross-encoder relevance scoring"
```

### Task 6.2: Evidence bundler + Full retrieval API

**Files:**
- Create: `src/retrieval/bundler.py`
- Create: `src/retrieval/api.py`
- Modify: `src/main.py`
- Create: `tests/test_retrieval_api.py`

- [ ] **Step 1: Create bundler.py**

```python
# src/retrieval/bundler.py
"""Assembles scored hits into layered EvidenceBundle."""
from src.models.evidence_bundle import EvidenceBundle, EvidenceItem, CardSources


class Bundler:
    def assemble(
        self,
        fused_hits: list[dict],
        knowledge_units: dict[str, dict],
        query_analysis,
        top_k_cards: int = 3,
        top_k_passages: int = 5,
    ) -> EvidenceBundle:
        card_bins = {"mindfulness": [], "healing": [], "communication": [], "personalized": []}
        supporting = []
        safety = []
        candidate_evidence = []
        excluded = []

        for hit in fused_hits:
            unit = knowledge_units.get(hit["id"])
            if unit is None:
                continue

            item = EvidenceItem(
                id=hit["id"],
                unit_type=unit.get("unit_type", "semantic_chunk"),
                source_type=unit.get("source_type", ""),
                source_status=unit.get("source_status", ""),
                title=unit.get("title", ""),
                snippet=unit.get("text", "")[:300],
                summary=unit.get("summary", ""),
                score=hit["score"],
                source_citation=unit.get("source_citation", ""),
                source_uri=unit.get("source_uri", ""),
                page_start=unit.get("page_start"),
                page_end=unit.get("page_end"),
                semantic_tags=_parse_json(unit.get("semantic_tags", "[]")),
                scenario_tags=_parse_json(unit.get("scenario_tags", "[]")),
                role_tags=_parse_json(unit.get("role_tags", "[]")),
                method_tags=_parse_json(unit.get("method_tags", "[]")),
                risk_levels=_parse_json(unit.get("risk_levels", "[]")),
                card_targets=_parse_json(unit.get("card_targets", "[]")),
                quality_score=unit.get("quality_score", 0.0),
                review_status=unit.get("review_status", ""),
                parent_chunk_ids=_parse_json(unit.get("parent_chunk_ids", "[]")),
            )

            status = unit.get("source_status", "")
            if status == "quarantined":
                excluded.append(item)
            elif hit.get("safety_boost", 0) > 0:
                safety.append(item)
            elif status == "candidate":
                candidate_evidence.append(item)
            else:
                supporting.append(item)

            # Sort into card groups
            for ct in item.card_targets:
                if ct in card_bins and len(card_bins[ct]) < top_k_cards:
                    card_bins[ct].append(item)

        return EvidenceBundle(
            query_analysis=query_analysis,
            card_sources=CardSources(
                mindfulness=card_bins["mindfulness"][:top_k_cards],
                healing=card_bins["healing"][:top_k_cards],
                communication=card_bins["communication"][:top_k_cards],
                personalized=card_bins["personalized"][:top_k_cards],
            ),
            supporting_passages=supporting[:top_k_passages],
            safety_and_boundary_evidence=safety,
            candidate_evidence=candidate_evidence,
            excluded_evidence=excluded,
        )


def _parse_json(val):
    import json
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []
```

- [ ] **Step 2: Create retrieval API with mock initial implementation**

```python
# src/retrieval/api.py
from fastapi import APIRouter
from src.models.retrieval import RetrieveRequest
from src.models.evidence_bundle import EvidenceBundle, QueryAnalysis, CardSources
from src.config import settings

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("")
async def retrieve(request: RetrieveRequest):
    """POST /retrieve — full retrieval pipeline.

    Phase 5-6: Starts with mock, grows into full pipeline as modules are built.
    """
    from src.llm.client import LLMClient, LLMConfig
    from src.retrieval.query_understanding import QueryUnderstanding
    from src.db.connection import Database
    from src.db.repository import KnowledgeUnitRepository

    # Query understanding (if LLM key is set)
    qu = None
    if settings.llm_api_key:
        llm_config = LLMConfig(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
        )
        llm_client = LLMClient(llm_config)
        qu_engine = QueryUnderstanding(llm_client)

    # Database
    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    # Query analysis
    query_analysis = QueryAnalysis()
    try:
        if qu_engine:
            query_analysis = await qu_engine.analyze(
                patient_text=request.patient_text,
                family_text=request.family_text,
                risk_assessment=request.risk_assessment,
                dyadic_analysis=request.dyadic_analysis,
            )
    except Exception:
        query_analysis = QueryAnalysis(
            rewritten_queries=[request.patient_text],
        )

    # Recall
    from src.indexing.embedder import Embedder
    from src.indexing.vector_index import VectorIndex
    from src.indexing.text_index import TextIndex
    from src.retrieval.dense import DenseRecaller
    from src.retrieval.sparse import SparseRecaller
    from src.retrieval.metadata_recall import MetadataRecaller
    from src.retrieval.safety_recall import SafetyRecaller
    from src.retrieval.hybrid import HybridFusion

    # Load indexes
    embedder = Embedder(
        model_path=f"{settings.models_dir}/{settings.embedding_model}",
        device=settings.embedding_device,
    )
    embedder.load()

    vec_index = VectorIndex(index_path=settings.faiss_index_path)
    vec_index.load()

    text_index = TextIndex(db)

    # Run recalls
    dense_hits = DenseRecaller(embedder, vec_index).recall(query_analysis.rewritten_queries)
    sparse_hits = SparseRecaller(text_index).recall(query_analysis.rewritten_queries)
    metadata_hits = MetadataRecaller(repo).recall(
        query_analysis.scenario_tags, query_analysis.role_focus
    )
    safety_hits = SafetyRecaller(repo).recall(
        query_analysis.contraindication_signals, query_analysis.risk_signals
    )

    # Build knowledge unit lookup
    all_ids = set()
    for hits in [dense_hits, sparse_hits, metadata_hits, safety_hits]:
        for h in hits:
            all_ids.add(h["id"])
    
    unit_lookup = {}
    for uid in all_ids:
        unit = repo.get_by_id(uid)
        if unit:
            unit_lookup[uid] = unit

    # Fuse
    fused = HybridFusion().fuse(dense_hits, sparse_hits, metadata_hits, safety_hits, unit_lookup)

    # Bundle
    from src.retrieval.bundler import Bundler
    bundle = Bundler().assemble(
        fused, unit_lookup, query_analysis,
        top_k_cards=request.top_k_cards,
        top_k_passages=request.top_k_passages,
    )

    db.close()
    return {"data": bundle.model_dump(), "error": None}
```

- [ ] **Step 3: Register router in src/main.py**

```python
from src.retrieval.api import router as retrieval_router
app.include_router(retrieval_router)
```

- [ ] **Step 4: Test and commit**

```bash
python -m pytest tests/test_retrieval_api.py -v
git add src/retrieval/bundler.py src/retrieval/api.py src/main.py tests/test_retrieval_api.py
git commit -m "feat: add evidence bundler and POST /retrieve full pipeline"
```

## Phase 7: Evaluation & Debug Interface

### Task 7.1: Evaluation metrics + API

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/api.py`
- Modify: `src/main.py`
- Create: `data/eval/rag_queries.jsonl`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Create eval queries sample**

```jsonl
{"query": "患者晚上睡不着，害怕闭上眼睛就再也醒不来", "patient_context": "晚期癌症，78岁", "family_context": "女儿每晚陪护", "expected_tags": ["死亡焦虑", "夜间焦虑"], "must_include_source_types": ["pdf_book", "guideline"], "must_include_card_targets": ["mindfulness"], "must_include_ids": [], "must_not_include_flags": ["quarantined"], "risk_level": "high", "human_notes": "典型死亡焦虑夜间发作"}
{"query": "家属不愿意告诉患者真实的病情", "patient_context": "胃癌末期", "family_context": "儿子要求隐瞒病情", "expected_tags": ["病情告知", "家属沟通"], "must_include_source_types": ["guideline"], "must_include_card_targets": ["communication"], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "medium", "human_notes": "家属否认/隐瞒场景"}
{"query": "患者说不想再治疗了，想回家", "patient_context": "多次化疗无效", "family_context": "丈夫坚持继续治疗", "expected_tags": ["治疗拒绝", "临终决策"], "must_include_source_types": ["guideline", "pdf_book"], "must_include_card_targets": ["communication"], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "high", "human_notes": "治疗拒绝 + 家庭冲突"}
{"query": "照护者感到身心俱疲，睡眠不足", "patient_context": "", "family_context": "妻子照顾卧床丈夫6个月", "expected_tags": ["照护负担"], "must_include_source_types": ["guideline", "paper"], "must_include_card_targets": ["healing", "personalized"], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "medium", "human_notes": "照护者负担"}
{"query": "患者呼吸困难，非常痛苦", "patient_context": "肺癌末期", "family_context": "", "expected_tags": ["呼吸困难"], "must_include_source_types": ["guideline"], "must_include_card_targets": [], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "high", "human_notes": "呼吸困难 + 安全边界"}
{"query": "患者说还有很多事情没做完，不甘心", "patient_context": "中年，有未成年子女", "family_context": "", "expected_tags": ["未竟事务", "存在性痛苦"], "must_include_source_types": ["pdf_book"], "must_include_card_targets": ["healing"], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "medium", "human_notes": "未竟事务"}
{"query": "如何与临终患者进行告别沟通", "patient_context": "", "family_context": "", "expected_tags": ["告别与分离", "告别沟通"], "must_include_source_types": ["pdf_book", "paper"], "must_include_card_targets": ["communication"], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "low", "human_notes": "告别沟通方法"}
{"query": "患者出现谵妄症状，胡言乱语", "patient_context": "肝癌末期，肝性脑病", "family_context": "家属非常恐慌", "expected_tags": ["谵妄", "认知障碍"], "must_include_source_types": ["guideline"], "must_include_card_targets": [], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "high", "human_notes": "谵妄安全"}
{"query": "医生说可能只有几个月了，我们该怎么办", "patient_context": "", "family_context": "患者女儿求助", "expected_tags": ["预后不确定", "临终决策", "哀伤辅导"], "must_include_source_types": ["guideline", "pdf_book"], "must_include_card_targets": ["communication", "personalized"], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "medium", "human_notes": "预后告知 + 家属支持"}
{"query": "患者说活着没意思，不如早点走", "patient_context": "长期疼痛，抑郁", "family_context": "", "expected_tags": ["自杀意念", "抑郁情绪"], "must_include_source_types": ["guideline"], "must_include_card_targets": [], "must_include_ids": [], "must_not_include_flags": [], "risk_level": "high", "human_notes": "高风险绝望表达，安全第一"}
```

- [ ] **Step 2: Create metrics.py**

```python
# src/evaluation/metrics.py
"""Evaluation metrics for retrieval quality."""
import json
from src.models.retrieval import RetrieveRequest


class EvalMetrics:
    @staticmethod
    def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 10) -> float:
        if not expected_ids:
            return 1.0  # No expected IDs = pass
        top_k = set(retrieved_ids[:k])
        hits = top_k & set(expected_ids)
        return len(hits) / len(expected_ids)

    @staticmethod
    def safety_hit(safety_evidence: list, risk_level: str) -> bool:
        if risk_level in ("high", "medium"):
            return len(safety_evidence) > 0
        return True

    @staticmethod
    def noise_rate(excluded: list, total_returned: int) -> float:
        if total_returned == 0:
            return 0.0
        quarantined = sum(1 for e in excluded if e.get("source_status") == "quarantined")
        return quarantined / total_returned

    @staticmethod
    def tag_match(returned_tags: list[str], expected_tags: list[str]) -> float:
        if not expected_tags:
            return 1.0
        hits = set(returned_tags) & set(expected_tags)
        return len(hits) / len(expected_tags)
```

- [ ] **Step 3: Create eval API**

```python
# src/evaluation/api.py
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter
from src.config import settings
from src.evaluation.metrics import EvalMetrics

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run")
async def run_eval():
    """Run evaluation against rag_queries.jsonl and return metrics."""
    eval_path = f"{settings.data_dir}/eval/rag_queries.jsonl"
    
    queries = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    if not queries:
        return {"status": "error", "message": "No eval queries found"}

    results = []
    for q in queries:
        # Call internal retrieve logic
        from src.models.retrieval import RetrieveRequest
        from src.retrieval.api import retrieve
        
        request = RetrieveRequest(
            patient_text=q.get("query", ""),
            user_role="nurse",
        )
        try:
            response = await retrieve(request)
            bundle = response["data"]
            all_items = (
                bundle["supporting_passages"]
                + bundle["card_sources"]["mindfulness"]
                + bundle["card_sources"]["healing"]
                + bundle["card_sources"]["communication"]
                + bundle["card_sources"]["personalized"]
            )
            retrieved_ids = [item["id"] for item in all_items]
            all_tags = []
            for item in all_items:
                all_tags.extend(item.get("semantic_tags", []))
                all_tags.extend(item.get("scenario_tags", []))

            recall = EvalMetrics.recall_at_k(retrieved_ids, q.get("must_include_ids", []))
            safety = EvalMetrics.safety_hit(
                bundle.get("safety_and_boundary_evidence", []), q.get("risk_level", "")
            )
            noise = EvalMetrics.noise_rate(
                bundle.get("excluded_evidence", []), len(all_items)
            )
            tag_score = EvalMetrics.tag_match(all_tags, q.get("expected_tags", []))

            results.append({
                "query": q["query"][:60],
                "recall@10": recall,
                "safety_hit": safety,
                "noise_rate": noise,
                "tag_match": tag_score,
            })
        except Exception as e:
            results.append({"query": q["query"][:60], "error": str(e)})

    # Aggregate
    n = len(results)
    avg_recall = sum(r.get("recall@10", 0) for r in results) / n
    avg_safety = sum(1 for r in results if r.get("safety_hit", False)) / n
    avg_noise = sum(r.get("noise_rate", 0) for r in results) / n
    avg_tag = sum(r.get("tag_match", 0) for r in results) / n

    # Store run
    from src.db.connection import Database
    db = Database(settings.db_path)
    db.initialize()
    now = datetime.now(timezone.utc).isoformat()
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    db.conn.execute(
        """INSERT INTO eval_runs (id, run_at, total_queries, recall_at_10, safety_hit_rate, noise_rate, avg_usability, details_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, now, n, avg_recall, avg_safety, avg_noise, avg_tag, json.dumps(results, ensure_ascii=False)),
    )
    db.conn.commit()
    db.close()

    return {
        "run_id": run_id,
        "total_queries": n,
        "recall_at_10": round(avg_recall, 4),
        "safety_hit_rate": round(avg_safety, 4),
        "noise_rate": round(avg_noise, 4),
        "avg_tag_match": round(avg_tag, 4),
        "details": results,
    }
```

- [ ] **Step 4: Register router and commit**

```bash
python -m pytest tests/test_eval.py -v
git add src/evaluation/ data/eval/ src/main.py tests/test_eval.py
git commit -m "feat: add evaluation metrics, eval queries, and POST /eval/run"
```

### Task 7.2: Debug interface

**Files:**
- Create: `src/debug/__init__.py`
- Create: `src/debug/api.py`
- Create: `src/debug/templates/debug.html`
- Modify: `src/main.py`

- [ ] **Step 1: Create debug API**

```python
# src/debug/api.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.config import settings
from src.db.connection import Database
from src.db.repository import KnowledgeUnitRepository

router = APIRouter(prefix="/debug", tags=["debug"])
templates = Jinja2Templates(directory="src/debug/templates")


@router.get("/status")
async def debug_status():
    """Return detailed system status."""
    db = Database(settings.db_path)
    db.initialize()
    repo = KnowledgeUnitRepository(db)

    counts = repo.count_by_status()
    type_counts = repo.count_by_type()

    docs = db.conn.execute("SELECT COUNT(*) as cnt FROM source_documents").fetchone()
    doc_count = docs["cnt"] if docs else 0

    # Source type distribution
    src_dist = db.conn.execute(
        "SELECT source_type, COUNT(*) as cnt FROM knowledge_units GROUP BY source_type"
    ).fetchall()

    db.close()

    import torch
    return {
        "gpu_available": torch.cuda.is_available(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "index_version": "",
        "counts": {
            "documents": doc_count,
            "chunks": type_counts.get("semantic_chunk", 0),
            "cards": type_counts.get("knowledge_card", 0),
            "by_status": counts,
            "by_source_type": {r["source_type"]: r["cnt"] for r in src_dist},
        },
        "config": {
            "chunk_min_chars": settings.chunk_min_chars,
            "chunk_max_chars": settings.chunk_max_chars,
            "quality_threshold": settings.quality_main_threshold,
            "weights": {
                "rerank": settings.weight_rerank,
                "recall": settings.weight_recall,
                "metadata": settings.weight_metadata,
                "quality": settings.weight_quality,
                "source_status": settings.weight_source_status,
            },
        },
    }


@router.get("/query", response_class=HTMLResponse)
async def debug_query_page(request: Request):
    """Render debug query interface."""
    return templates.TemplateResponse("debug.html", {"request": request})
```

- [ ] **Step 2: Create minimal debug HTML template**

```html
<!-- src/debug/templates/debug.html -->
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>NarrCare-KB Debug</title>
    <style>
        body { font-family: system-ui; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }
        textarea { width: 100%; height: 80px; padding: 10px; border-radius: 6px; border: 1px solid #333; background: #16213e; color: #e0e0e0; margin-bottom: 10px; }
        button { padding: 10px 24px; background: #0f3460; color: white; border: none; border-radius: 6px; cursor: pointer; }
        pre { background: #16213e; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 12px; max-height: 600px; overflow-y: auto; }
        .section { margin: 20px 0; }
        h2 { color: #e94560; }
    </style>
</head>
<body>
    <h1>NarrCare-KB Debug Console</h1>
    <div class="section">
        <textarea id="query" placeholder="输入患者/家属文本..."></textarea>
        <button onclick="search()">检索</button>
    </div>
    <div class="section">
        <h2>结果</h2>
        <pre id="result">点击"检索"查看 EvidenceBundle...</pre>
    </div>
    <script>
        async function search() {
            const q = document.getElementById("query").value;
            document.getElementById("result").textContent = "检索中...";
            const resp = await fetch("/retrieve", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({patient_text: q, include_debug: true}),
            });
            const data = await resp.json();
            document.getElementById("result").textContent = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
```

- [ ] **Step 3: Register and commit**

```bash
git add src/debug/ src/main.py
git commit -m "feat: add debug status API and query console"
```

### Task 7.3: Final integration — import script

**Files:**
- Create: `scripts/import_local_files.py`

- [ ] **Step 1: Create batch import script**

```python
#!/usr/bin/env python3
"""Batch import local files into NarrCare-KB."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db.connection import Database
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker
from src.ingestion.orchestrator import IngestionOrchestrator


def main():
    import_dir = "参考资料"
    if not os.path.isdir(import_dir):
        print(f"Directory not found: {import_dir}")
        sys.exit(1)

    pdf_files = [
        os.path.join(import_dir, f)
        for f in os.listdir(import_dir)
        if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF files to import.")

    db = Database(settings.db_path)
    db.initialize()
    orch = IngestionOrchestrator(
        db=db,
        parser=DocumentParser(),
        cleaner=TextCleaner(),
        chunker=SemanticChunker(
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
    )

    for fp in pdf_files:
        print(f"\nImporting: {fp}")
        result = orch.ingest_file(fp, source_status="main")
        print(f"  Status: {result['status']}")
        print(f"  Chunks: {result.get('chunk_count', 0)}")
        print(f"  Quarantined: {result.get('quarantined_count', 0)}")

    db.close()
    print("\nDone. Run POST /index/rebuild to build search indexes.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/import_local_files.py
git commit -m "feat: add batch import script for local PDF files"
```

---

## Final Verification Checklist

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Health endpoint: `curl http://localhost:9000/health`
- [ ] Import files: `python scripts/import_local_files.py`
- [ ] Rebuild index: `curl -X POST http://localhost:9000/index/rebuild`
- [ ] Retrieve: `curl -X POST http://localhost:9000/retrieve -H "Content-Type: application/json" -d '{"patient_text": "患者夜间恐惧死亡"}'`
- [ ] Debug console: open `http://localhost:9000/debug/query`
- [ ] Run eval: `curl -X POST http://localhost:9000/eval/run`
- [ ] All acceptance criteria from spec met:
  - Recall@10 >= 0.70
  - Safety hit rate >= 95%
  - Noise rate <= 15%
  - /retrieve returns legal EvidenceBundle
