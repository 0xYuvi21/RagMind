"""
ConversationService — orchestrates repositories, the per-chat vector store,
and the LangGraph agent. This is where chat isolation actually gets wired
together: `ask()` builds the retrieval tool from THIS conversation's
VectorStore only, and invokes the agent with thread_id=str(conversation_id),
so both the documents and the memory are scoped to one chat.

Ownership enforcement lives here (`_get_owned_conversation`), not in the
repository layer — a conversation_id that exists but belongs to another user
is treated identically to one that doesn't exist (return None), so routers
can turn both into a 404 without leaking whether the id exists at all.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from Backend.db_models import Conversation, Message
from Backend.repositories import SqlAlchemyConversationRepository, SqlAlchemyMessageRepository
from RAG.vector_store import VectorStoreFactory
from Retrieve.llmquery import build_chat_agent, build_retrieval_tool

_SOURCES_PATTERN = re.compile(
    r"Sources:\s*(.*?)(?:\n\nFollow-up:|$)", re.IGNORECASE | re.DOTALL
)


class ConversationNotFoundError(Exception):
    """Raised when a conversation doesn't exist OR doesn't belong to the
    caller — routers map this to a 404 either way."""


class ConversationService:
    def __init__(
        self,
        conversation_repository: SqlAlchemyConversationRepository,
        message_repository: SqlAlchemyMessageRepository,
        vector_store_factory: VectorStoreFactory,
        llm_provider,
        checkpoint_backend,
        uploads_dir: str,
    ):
        self._conversations = conversation_repository
        self._messages = message_repository
        self._vector_store_factory = vector_store_factory
        self._llm_provider = llm_provider
        self._checkpoint_backend = checkpoint_backend
        self._uploads_dir = Path(uploads_dir)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create(self, user_id: int, title: Optional[str]) -> Conversation:
        return self._conversations.create(user_id=user_id, title=title or "New Conversation")

    def list_for_user(self, user_id: int) -> List[Conversation]:
        return self._conversations.list_for_user(user_id)

    def get_owned(self, conversation_id: int, user_id: int) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    def rename(self, conversation_id: int, user_id: int, title: str) -> Conversation:
        self.get_owned(conversation_id, user_id)
        return self._conversations.update_title(conversation_id, title)

    def delete(self, conversation_id: int, user_id: int) -> None:
        self.get_owned(conversation_id, user_id)
        self._vector_store_factory.delete_for_chat(str(conversation_id))
        self._conversations.delete(conversation_id)

    def list_messages(self, conversation_id: int, user_id: int) -> List[Message]:
        self.get_owned(conversation_id, user_id)
        return self._messages.list_for_conversation(conversation_id)

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest_upload(
        self, conversation_id: int, user_id: int, file_content: bytes, file_name: str
    ) -> int:
        """Ingest an uploaded file into THIS conversation's isolated store,
        immediately queryable — no server restart needed, since the store
        instance is cached and reused by build_retrieval_tool() on the very
        next ask() call."""
        self.get_owned(conversation_id, user_id)

        # Persist to disk too, so the VLM read_image tool can resolve a path
        # for any image the agent decides to look at more closely.
        chat_dir = self._uploads_dir / str(user_id) / str(conversation_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        safe_name = file_name.replace("/", "_").replace("\\", "_")
        (chat_dir / safe_name).write_bytes(file_content)

        store = self._vector_store_factory.get_or_create(str(conversation_id))
        chunks_added = store.add_file_stream(
            file_content=file_content,
            file_name=file_name,
            source_tag="user_upload",
            image_output_dir=str(chat_dir),
        )
        self._conversations.touch(conversation_id)
        return chunks_added

    # ── Chat ──────────────────────────────────────────────────────────────

    def ask(self, conversation_id: int, user_id: int, query: str) -> Tuple[str, List[str]]:
        self.get_owned(conversation_id, user_id)

        store = self._vector_store_factory.get_or_create(str(conversation_id))
        tools = [build_retrieval_tool(store)]
        tools.extend(self._optional_vlm_tool())

        agent = build_chat_agent(
            llm_provider=self._llm_provider,
            tools=tools,
            checkpointer=self._checkpoint_backend.get_saver(),
        )

        config = {"configurable": {"thread_id": str(conversation_id)}}
        result = agent.invoke({"messages": [("user", query)]}, config=config)
        answer = result["messages"][-1].content

        sources = self._extract_sources(answer)

        self._messages.add(conversation_id, role="user", content=query)
        self._messages.add(conversation_id, role="assistant", content=answer, sources=sources)
        self._conversations.touch(conversation_id)

        return answer, sources

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _optional_vlm_tool() -> list:
        """The VLM tool depends on a locally-running Ollama instance, which
        isn't guaranteed to be available in every environment (see
        CLAUDE.md). Import lazily and skip it rather than failing chat
        entirely if VLM/imagellm.py's dependencies aren't usable."""
        try:
            from VLM.imagellm import read_image

            return [read_image]
        except Exception:
            return []

    @staticmethod
    def _extract_sources(answer: str) -> List[str]:
        sources: set[str] = set()
        match = _SOURCES_PATTERN.search(answer)
        if match:
            for raw_line in match.group(1).split("\n"):
                cleaned = raw_line.strip("-* \t")
                if cleaned:
                    sources.add(cleaned)
        return sorted(sources)
