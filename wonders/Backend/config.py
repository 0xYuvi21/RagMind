"""
Centralized application settings, loaded from environment variables / .env.

This is the single place that reads os.environ — every other module receives
configuration through constructor injection instead of reading env vars
itself (replacing the previous pattern of GROQ_API_KEY/UNSTRUCTURED_API_KEY
being read ad-hoc inside RAG/ingest.py and Retrieve/llmquery.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
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

    # ── Retrieval / chunking ──────────────────────────────────────────────
    retrieval_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 100


def get_settings() -> Settings:
    settings = Settings()
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    return settings
