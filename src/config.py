"""Global configuration via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KB_")

    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    models_dir: str = "models"
    data_dir: str = "data"
    db_path: str = "data/db/kb.sqlite"
    faiss_index_path: str = "data/index/faiss.index"
    embedding_model: str = "Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen3-Reranker-0.6B"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60
    dense_top_k: int = 50
    sparse_top_k: int = 30
    metadata_top_k: int = 20
    safety_top_k: int = 10
    default_top_k_cards: int = 3
    default_top_k_passages: int = 5
    weight_rerank: float = 0.55
    weight_recall: float = 0.20
    weight_metadata: float = 0.10
    weight_quality: float = 0.10
    weight_source_status: float = 0.05
    chunk_min_chars: int = 300
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100
    quality_main_threshold: float = 0.5


settings = Settings()
