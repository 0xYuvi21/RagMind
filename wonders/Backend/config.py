"""
Centralized application settings, loaded from config.yaml / environment
variables / .env.

This is the single place that reads os.environ (and config.yaml) — every
other module receives configuration through constructor injection instead of
reading env vars or the yaml file itself (replacing the previous pattern of
GROQ_API_KEY/UNSTRUCTURED_API_KEY being read ad-hoc inside RAG/ingest.py and
Retrieve/llmquery.py).

Which model/provider to use (Groq model name, embedding model, VLM/OCR
backend, retrieval/chunking knobs) lives in config.yaml at the project root —
see that file's header comment. Secrets and environment-specific paths (JWT
key, DATABASE_URL, CHROMA_PERSIST_DIR, ...) stay in .env. Precedence (highest
first): constructor kwargs (tests) > env vars > .env > config.yaml > field
default below — so config.yaml is the easiest place to change a model, but an
env var still wins if one is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple, Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        yaml_file=str(PROJECT_ROOT / "config.yaml"),
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # ── Auth ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24h access token, no refresh token (see SYSTEM_DESIGN.md)

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"

    # ── LLM ───────────────────────────────────────────────────────────────
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector store ──────────────────────────────────────────────────────
    chroma_persist_dir: str = str(DATA_DIR / "chroma")

    # ── Checkpointing (LangGraph short-term memory) ──────────────────────
    checkpointer_backend: Literal["sqlite", "memory"] = "sqlite"
    checkpoint_db_path: str = str(DATA_DIR / "checkpoints.sqlite")

    # ── Uploads ───────────────────────────────────────────────────────────
    uploads_dir: str = str(DATA_DIR / "uploads")

    # ── VLM (Ollama) ──────────────────────────────────────────────────────
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_vlm_model: str = "richardyoung/smolvlm2-2.2b-instruct:latest"

    # ── OCR (exact text extraction from images — see RAG/ocr.py) ─────────
    ocr_enabled: bool = True
    ocr_provider: Literal["rapidocr"] = "rapidocr"

    # ── Kafka (async document ingestion — see docker-compose.yml,
    # Backend/kafka_producer.py, Backend/ingestion_worker.py) ────────────
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_ingestion_topic: str = "document-ingestion"
    kafka_consumer_group: str = "ragmind-ingestion-workers"

    # ── Retrieval / chunking ──────────────────────────────────────────────
    retrieval_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 100

    # ── Audit logging (separate PostgreSQL database — see
    # Backend/audit_db.py, Backend/audit_models.py). Left blank on purpose:
    # fill in real credentials to enable it. Until then, Backend/deps.py
    # wires in NullAuditLogger and every action proceeds exactly as before,
    # just without an audit trail. ────────────────────────────────────────
    audit_postgres_host: str = ""
    audit_postgres_port: int = 5432
    audit_postgres_db: str = ""
    audit_postgres_user: str = ""
    audit_postgres_password: str = ""

    @property
    def audit_database_url(self) -> str | None:
        """None (audit logging disabled) unless host/db/user are all set."""
        if not (self.audit_postgres_host and self.audit_postgres_db and self.audit_postgres_user):
            return None
        return (
            f"postgresql+psycopg2://{self.audit_postgres_user}:{self.audit_postgres_password}"
            f"@{self.audit_postgres_host}:{self.audit_postgres_port}/{self.audit_postgres_db}"
        )


def get_settings() -> Settings:
    settings = Settings()
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    return settings
