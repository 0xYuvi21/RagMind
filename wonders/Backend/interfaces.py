"""
Abstractions (Protocols) for every swappable concern in the system.

Concrete implementations live next to their domain (RAG/ingest.py,
RAG/vector_store.py, Retrieve/llmquery.py, Backend/repositories.py) and are
wired together only in Backend/deps.py (the composition root). Routes/services
depend on these Protocols — never on a concrete class or a module-level
singleton like the old `pipeline`/`_store` in Retrieve/llmquery.py.

Method names on DocumentParser/VectorStoreBackend mirror the existing
DocumentIngestor/VectorStore APIs (`ingest`, `ingest_file_stream`,
`similarity_search`, ...) so those classes satisfy these protocols with no
renaming.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from Backend.db_models import Conversation, IngestionJob, Message, User


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Strategy for turning text into vectors."""

    def get_embeddings(self) -> Embeddings: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Strategy for the chat model used by the agent."""

    def get_chat_model(self) -> BaseChatModel: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Strategy for turning a raw file into LangChain Documents (pre-chunking)."""

    def ingest(self, file_path: str) -> List[Document]: ...

    def ingest_file_stream(self, file_content: bytes, file_name: str) -> List[Document]: ...


@runtime_checkable
class VectorStoreBackend(Protocol):
    """A single, isolated document store (one per conversation)."""

    def add_file_stream(
        self,
        file_content: bytes,
        file_name: str,
        source_tag: Optional[str] = None,
        image_output_dir: Optional[str] = None,
    ) -> int: ...

    def similarity_search(self, query: str, k: int = 5) -> List[Document]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class VectorStoreFactoryProtocol(Protocol):
    """Resolves the isolated VectorStoreBackend for a given chat/conversation id."""

    def get_or_create(self, chat_id: str) -> VectorStoreBackend: ...

    def delete_for_chat(self, chat_id: str) -> None: ...


@runtime_checkable
class CheckpointBackend(Protocol):
    """Strategy for LangGraph's conversational state persistence."""

    def get_saver(self) -> BaseCheckpointSaver: ...


class UserRepository(Protocol):
    def create(self, email: str, hashed_password: str) -> User: ...

    def get_by_email(self, email: str) -> Optional[User]: ...

    def get_by_id(self, user_id: int) -> Optional[User]: ...


class ConversationRepository(Protocol):
    def create(self, user_id: int, title: str) -> Conversation: ...

    def list_for_user(self, user_id: int) -> List[Conversation]: ...

    def get(self, conversation_id: int) -> Optional[Conversation]: ...

    def update_title(self, conversation_id: int, title: str) -> Optional[Conversation]: ...

    def touch(self, conversation_id: int) -> None: ...

    def delete(self, conversation_id: int) -> None: ...


class MessageRepository(Protocol):
    def add(
        self, conversation_id: int, role: str, content: str, sources: Optional[List[str]] = None
    ) -> Message: ...

    def list_for_conversation(self, conversation_id: int) -> List[Message]: ...


class IngestionJobRepository(Protocol):
    """Tracks the lifecycle of an async document-ingestion job (see
    Backend/db_models.py's IngestionJob, Backend/kafka_producer.py)."""

    def create(self, conversation_id: int, file_name: str, stored_path: str) -> IngestionJob: ...

    def get(self, job_id: int) -> Optional[IngestionJob]: ...

    def mark_processing(self, job_id: int) -> None: ...

    def mark_done(self, job_id: int, chunks_added: int) -> None: ...

    def mark_failed(self, job_id: int, error_message: str) -> None: ...


@runtime_checkable
class IngestionQueueProducer(Protocol):
    """Strategy for handing an ingestion job off to be processed
    asynchronously. The real implementation (Backend/kafka_producer.py)
    publishes to Kafka; Backend/ingestion_worker.py's consumer does the
    actual parsing/embedding in a separate process. `stored_path` is a path
    on shared disk, not the file bytes — keeps queue messages small."""

    def publish(
        self,
        job_id: int,
        conversation_id: int,
        user_id: int,
        file_name: str,
        stored_path: str,
    ) -> None: ...


@runtime_checkable
class AuditLogger(Protocol):
    """Strategy for recording every action in the system to durable storage
    (Backend/audit_models.py — one table per action category in a dedicated
    Postgres database, see Backend/audit_db.py). Every method is a single,
    append-only "record this event" call — never a query — so callers
    (AuthService, ConversationService, the ingestion worker, the LLM/VLM
    call sites) can depend on this narrow interface without pulling in
    persistence details, per the Dependency Inversion Principle.

    Real implementation: Backend/audit_logger.py's SqlAlchemyAuditLogger.
    Fallback: NullAuditLogger, used when Postgres credentials aren't
    configured yet (see config.yaml/.env's audit_database_url) — mirrors the
    existing defensive pattern for the optional VLM tool and Kafka producer:
    a missing dependency degrades logging, not the whole request."""

    def log_auth_event(
        self, action: str, email: str, success: bool, user_id: Optional[int] = None, detail: Optional[str] = None
    ) -> None: ...

    def log_conversation_event(
        self, action: str, conversation_id: int, user_id: int, detail: Optional[str] = None
    ) -> None: ...

    def log_message_event(
        self, conversation_id: int, user_id: int, role: str, content_length: int
    ) -> None: ...

    def log_ingestion_event(
        self,
        conversation_id: int,
        user_id: int,
        job_id: int,
        file_name: str,
        status: str,
        chunks_added: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None: ...

    def log_llm_event(
        self,
        conversation_id: int,
        user_id: Optional[int],
        provider: str,
        model: str,
        success: bool,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None: ...

    def log_vlm_event(
        self,
        model: str,
        image_path: str,
        success: bool,
        conversation_id: Optional[int] = None,
        user_id: Optional[int] = None,
        latency_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None: ...
