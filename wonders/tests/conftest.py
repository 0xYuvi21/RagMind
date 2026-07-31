"""
Shared pytest fixtures.

Every fixture here overrides the exact same Depends functions
Backend/deps.py exposes to production code (get_settings_cached,
get_db_session, get_vector_store_factory, get_llm_provider,
get_checkpoint_backend) — so tests exercise the real composition root, just
wired to a temp SQLite DB, a temp Chroma dir, an in-memory checkpointer, and
the fakes in tests/fakes.py instead of live infrastructure.
"""

from __future__ import annotations

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from Backend import deps
from Backend.backend import app
from Backend.config import Settings
from Backend.db import build_engine, build_session_factory, init_db
from Backend.repositories import SqlAlchemyIngestionJobRepository
from RAG.vector_store import VectorStoreFactory
from Retrieve.checkpointer import InMemoryCheckpointBackend
from tests.fakes import FakeIngestionProducer, FakeLLMProvider, make_fake_embeddings, make_final_answer


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        chroma_persist_dir=str(tmp_path / "chroma"),
        uploads_dir=str(tmp_path / "uploads"),
        checkpointer_backend="memory",
        jwt_secret_key="test-secret-key-at-least-32-bytes-long-for-hs256",
        jwt_expire_minutes=60,
        groq_api_key=None,
    )


@pytest.fixture
def vector_store_factory(test_settings):
    return VectorStoreFactory(
        persist_dir=test_settings.chroma_persist_dir,
        embedding_fn=make_fake_embeddings(),
        chunk_size=test_settings.chunk_size,
        chunk_overlap=test_settings.chunk_overlap,
    )


@pytest.fixture
def checkpoint_backend():
    return InMemoryCheckpointBackend()


@pytest.fixture
def llm_script():
    """Mutable holder: tests set `llm_script["responses"]` before hitting
    /chat, and can read `llm_script["capture"]` afterward to inspect exactly
    which messages the fake model was asked to generate from."""
    return {
        "responses": [make_final_answer("Answer:\nHello.\n\nSources:\n\nFollow-up:\nAnything else?")],
        "capture": [],
    }


@pytest.fixture
def client(test_settings, vector_store_factory, checkpoint_backend, llm_script):
    engine = build_engine(test_settings.database_url)
    init_db(engine)
    session_factory = build_session_factory(engine)

    def override_get_settings():
        return test_settings

    def override_get_db_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    def override_vector_store_factory():
        return vector_store_factory

    def override_checkpoint_backend():
        return checkpoint_backend

    def override_llm_provider():
        return FakeLLMProvider(list(llm_script["responses"]), capture=llm_script["capture"])

    def override_ingestion_producer(session=Depends(deps.get_db_session)):
        return FakeIngestionProducer(
            job_repository=SqlAlchemyIngestionJobRepository(session),
            vector_store_factory=vector_store_factory,
        )

    app.dependency_overrides[deps.get_settings_cached] = override_get_settings
    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    app.dependency_overrides[deps.get_vector_store_factory] = override_vector_store_factory
    app.dependency_overrides[deps.get_checkpoint_backend] = override_checkpoint_backend
    app.dependency_overrides[deps.get_llm_provider] = override_llm_provider
    app.dependency_overrides[deps.get_ingestion_producer] = override_ingestion_producer

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def register_and_login(client: TestClient, email: str, password: str = "password123") -> dict:
    """Returns Authorization headers for a freshly registered+logged-in user."""
    client.post("/auth/register", json={"email": email, "password": password})
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers(client):
    return register_and_login(client, "student@example.com")
