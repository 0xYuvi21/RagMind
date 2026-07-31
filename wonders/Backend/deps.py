"""
Composition root.

Every dependency the routes need is resolved here via FastAPI's `Depends`,
built from injected Settings — nothing is a module-level singleton computed
at import time (the pattern this rebuild specifically eliminates: the old
`from llmquery import pipeline, _store`). Tests override these same functions
via `app.dependency_overrides` (see tests/conftest.py) to swap in a temp
SQLite DB, a temp Chroma dir, and fakes for the LLM/embeddings/checkpointer —
exercising the exact same wiring path production uses.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Generator, Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from Backend.audit_db import build_audit_engine, build_audit_session_factory, init_audit_db
from Backend.audit_logger import NullAuditLogger, SqlAlchemyAuditLogger
from Backend.config import Settings
from Backend.config import get_settings as _load_settings
from Backend.db import build_engine, build_session_factory, init_db
from Backend.interfaces import AuditLogger, IngestionQueueProducer
from Backend.kafka_producer import KafkaIngestionProducer
from Backend.repositories import (
    SqlAlchemyConversationRepository,
    SqlAlchemyIngestionJobRepository,
    SqlAlchemyMessageRepository,
    SqlAlchemyUserRepository,
)
from RAG.embeddings import HuggingFaceEmbeddingProvider
from RAG.ocr import OcrProvider, RapidOcrProvider
from RAG.vector_store import VectorStoreFactory
from Retrieve.checkpointer import build_checkpoint_backend
from Retrieve.llmquery import GroqLLMProvider


@lru_cache
def get_settings_cached() -> Settings:
    return _load_settings()


@lru_cache
def _get_engine():
    return build_engine(get_settings_cached().database_url)


@lru_cache
def _get_session_factory():
    engine = _get_engine()
    init_db(engine)
    return build_session_factory(engine)


def init_app_db() -> None:
    """Create tables if they don't exist yet. Called once from backend.py's
    lifespan startup hook."""
    init_db(_get_engine())
    if get_settings_cached().audit_database_url:
        init_audit_db(_get_audit_engine())


def get_db_session() -> Generator[Session, None, None]:
    session_factory = _get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_user_repository(session: Session = Depends(get_db_session)) -> SqlAlchemyUserRepository:
    return SqlAlchemyUserRepository(session)


def get_conversation_repository(
    session: Session = Depends(get_db_session),
) -> SqlAlchemyConversationRepository:
    return SqlAlchemyConversationRepository(session)


def get_message_repository(
    session: Session = Depends(get_db_session),
) -> SqlAlchemyMessageRepository:
    return SqlAlchemyMessageRepository(session)


def get_ingestion_job_repository(
    session: Session = Depends(get_db_session),
) -> SqlAlchemyIngestionJobRepository:
    return SqlAlchemyIngestionJobRepository(session)


@lru_cache
def _get_audit_engine():
    return build_audit_engine(get_settings_cached().audit_database_url)


@lru_cache
def _get_audit_session_factory():
    return build_audit_session_factory(_get_audit_engine())


@lru_cache
def get_audit_logger() -> AuditLogger:
    """NullAuditLogger unless AUDIT_POSTGRES_HOST/DB/USER are all configured
    (see Settings.audit_database_url) — a blank audit database config is the
    documented default until real Postgres credentials are filled in (see
    CLAUDE.md), not an error."""
    settings = get_settings_cached()
    if not settings.audit_database_url:
        return NullAuditLogger()
    return SqlAlchemyAuditLogger(_get_audit_session_factory())


@lru_cache
def get_ingestion_producer() -> IngestionQueueProducer:
    settings = get_settings_cached()
    return KafkaIngestionProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.kafka_ingestion_topic,
    )


@lru_cache
def _get_embedding_provider() -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider(get_settings_cached().embedding_model_name)


@lru_cache
def _get_ocr_provider() -> Optional[OcrProvider]:
    """Which OCR backend (if any) is active is controlled entirely by
    config.yaml (ocr_enabled / ocr_provider) — no code change needed to
    disable OCR or swap in a different backend later."""
    settings = get_settings_cached()
    if not settings.ocr_enabled:
        return None
    if settings.ocr_provider == "rapidocr":
        return RapidOcrProvider()
    return None


@lru_cache
def get_vector_store_factory() -> VectorStoreFactory:
    settings = get_settings_cached()
    return VectorStoreFactory(
        persist_dir=settings.chroma_persist_dir,
        embedding_fn=_get_embedding_provider().get_embeddings(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        ocr_provider=_get_ocr_provider(),
    )


@lru_cache
def get_checkpoint_backend():
    settings = get_settings_cached()
    return build_checkpoint_backend(settings.checkpointer_backend, settings.checkpoint_db_path)


def get_llm_provider(settings: Settings = Depends(get_settings_cached)) -> GroqLLMProvider:
    return GroqLLMProvider(api_key=settings.groq_api_key, model=settings.groq_model)


# `get_settings` re-exported so other modules (e.g. auth_dependencies.py) can
# Depends() on the same cached settings function used everywhere else.
get_settings = get_settings_cached

# `get_session_factory` re-exported (was `_get_session_factory`) so
# Backend/ingestion_worker.py — a standalone process, not a FastAPI route —
# can build its own DB sessions from the exact same engine/tables the web
# process uses, without reaching into a "private" name.
get_session_factory = _get_session_factory
