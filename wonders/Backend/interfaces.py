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

from Backend.db_models import Conversation, Message, User


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
