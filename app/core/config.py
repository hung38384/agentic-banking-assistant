from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Banking Assistant"
    environment: str = "local"
    api_key: str = "dev-secret"

    llm_provider: str = Field(default="mock", description="mock, gemini, groq, or openai")
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    ragas_judge_provider: str = "groq"
    ragas_app_provider: str = "mock"
    ragas_openai_judge_model: str = "gpt-4o-mini"
    ragas_openai_embedding_model: str = "text-embedding-3-small"

    vector_backend: str = "chroma"
    chroma_dir: str = ".chroma"
    collection_name: str = "banking_docs"
    top_k: int = 4
    max_context_chars: int = 4500
    hybrid_vector_weight: float = 0.65
    hybrid_bm25_weight: float = 0.35
    rerank_top_n: int = 8
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.94
    hitl_enabled: bool = True
    memory_persistence_enabled: bool = True
    memory_path: Path = Path("data/runtime/conversation_memory.json")
    memory_persist_requires_consent: bool = True

    otel_enabled: bool = False
    otel_service_name: str = "agentic-banking-assistant"
    otel_exporter_otlp_endpoint: str | None = None

    data_dir: Path = Path("data")
    docs_dir: Path = Path("data/docs")
    policy_path: Path = Path("data/policies/guardrails.yml")
    dataset_version_path: Path = Path("data/eval/dataset_version.json")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
