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
import time
from pathlib import Path
from typing import List, Optional, Tuple

from Backend.audit_logger import NullAuditLogger
from Backend.db_models import Conversation, IngestionJob, Message
from Backend.interfaces import AuditLogger
from Backend.repositories import (
    SqlAlchemyConversationRepository,
    SqlAlchemyIngestionJobRepository,
    SqlAlchemyMessageRepository,
)
from RAG.ingest import SUPPORTED_EXTENSIONS
from RAG.vector_store import VectorStoreFactory
from Retrieve.llmquery import build_chat_agent, build_retrieval_tool

_SOURCES_PATTERN = re.compile(
    r"Sources:\s*(.*?)(?:\n\nFollow-up:|$)", re.IGNORECASE | re.DOTALL
)


class ConversationNotFoundError(Exception):
    """Raised when a conversation doesn't exist OR doesn't belong to the
    caller — routers map this to a 404 either way."""


class IngestionJobNotFoundError(Exception):
    """Raised when a job id doesn't exist OR its conversation doesn't belong
    to the caller — same "don't confirm existence" rationale as
    ConversationNotFoundError."""


class ConversationService:
    def __init__(
        self,
        conversation_repository: SqlAlchemyConversationRepository,
        message_repository: SqlAlchemyMessageRepository,
        vector_store_factory: VectorStoreFactory,
        llm_provider,
        checkpoint_backend,
        uploads_dir: str,
        ingestion_job_repository: SqlAlchemyIngestionJobRepository,
        ingestion_producer,
        ollama_url: Optional[str] = None,
        ollama_vlm_model: Optional[str] = None,
        audit_logger: AuditLogger | None = None,
    ):
        self._conversations = conversation_repository
        self._messages = message_repository
        self._vector_store_factory = vector_store_factory
        self._llm_provider = llm_provider
        self._checkpoint_backend = checkpoint_backend
        self._uploads_dir = Path(uploads_dir)
        self._ingestion_jobs = ingestion_job_repository
        self._ingestion_producer = ingestion_producer
        self._ollama_url = ollama_url
        self._ollama_vlm_model = ollama_vlm_model
        self._audit = audit_logger or NullAuditLogger()

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create(self, user_id: int, title: Optional[str]) -> Conversation:
        conversation = self._conversations.create(user_id=user_id, title=title or "New Conversation")
        self._audit.log_conversation_event("create", conversation.id, user_id)
        return conversation

    def list_for_user(self, user_id: int) -> List[Conversation]:
        return self._conversations.list_for_user(user_id)

    def get_owned(self, conversation_id: int, user_id: int) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    def rename(self, conversation_id: int, user_id: int, title: str) -> Conversation:
        self.get_owned(conversation_id, user_id)
        conversation = self._conversations.update_title(conversation_id, title)
        self._audit.log_conversation_event(
            "rename", conversation_id, user_id, detail=f"New title: {title}"
        )
        return conversation

    def delete(self, conversation_id: int, user_id: int) -> None:
        self.get_owned(conversation_id, user_id)
        self._vector_store_factory.delete_for_chat(str(conversation_id))
        self._conversations.delete(conversation_id)
        self._audit.log_conversation_event("delete", conversation_id, user_id)

    def list_messages(self, conversation_id: int, user_id: int) -> List[Message]:
        self.get_owned(conversation_id, user_id)
        return self._messages.list_for_conversation(conversation_id)

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest_upload(
        self, conversation_id: int, user_id: int, file_content: bytes, file_name: str
    ) -> IngestionJob:
        """Queue an uploaded file for async ingestion instead of parsing/
        embedding it inline: the actual work (parse -> chunk -> embed ->
        Chroma) happens in a separate process (Backend/ingestion_worker.py),
        consuming a Kafka message this method publishes, so many concurrent
        uploads don't all compete for CPU/embedding-model time on the one
        request-handling process. Returns immediately with a queued
        IngestionJob; callers poll get_ingestion_job() for completion (see
        Backend/conversations_router.py's GET .../uploads/{job_id})."""
        self.get_owned(conversation_id, user_id)

        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {ext}")

        # Persist to disk first — the worker reads bytes from here (Kafka
        # carries only a job reference, not the file itself), and it's also
        # what lets the VLM read_image tool resolve a path for any image the
        # agent decides to look at more closely.
        chat_dir = self._uploads_dir / str(user_id) / str(conversation_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        safe_name = file_name.replace("/", "_").replace("\\", "_")
        stored_path = chat_dir / safe_name
        stored_path.write_bytes(file_content)

        job = self._ingestion_jobs.create(conversation_id, file_name, str(stored_path))

        try:
            self._ingestion_producer.publish(
                job_id=job.id,
                conversation_id=conversation_id,
                user_id=user_id,
                file_name=file_name,
                stored_path=str(stored_path),
            )
        except Exception as exc:
            # Queue unreachable (e.g. Kafka down) — report it on the job
            # itself rather than a 500, same defensive posture as the
            # optional VLM tool: a broken dependency degrades one feature,
            # not the whole request.
            self._ingestion_jobs.mark_failed(job.id, f"Could not queue ingestion job: {exc}")
            failed_job = self._ingestion_jobs.get(job.id)
            self._audit.log_ingestion_event(
                conversation_id,
                user_id,
                job.id,
                file_name,
                status=failed_job.status,
                error_message=failed_job.error_message,
            )
            return failed_job

        self._conversations.touch(conversation_id)
        self._audit.log_ingestion_event(
            conversation_id, user_id, job.id, file_name, status=job.status
        )
        return job

    def get_ingestion_job(self, conversation_id: int, user_id: int, job_id: int) -> IngestionJob:
        self.get_owned(conversation_id, user_id)
        job = self._ingestion_jobs.get(job_id)
        if job is None or job.conversation_id != conversation_id:
            raise IngestionJobNotFoundError(job_id)
        return job

    # ── Chat ──────────────────────────────────────────────────────────────

    def ask(self, conversation_id: int, user_id: int, query: str) -> Tuple[str, List[str]]:
        self.get_owned(conversation_id, user_id)

        store = self._vector_store_factory.get_or_create(str(conversation_id))
        tools = [build_retrieval_tool(store)]
        tools.extend(self._optional_vlm_tool(conversation_id, user_id))

        agent = build_chat_agent(
            llm_provider=self._llm_provider,
            tools=tools,
            checkpointer=self._checkpoint_backend.get_saver(),
        )

        config = {"configurable": {"thread_id": str(conversation_id)}}
        model_name = getattr(self._llm_provider, "_model", "unknown")
        started = time.monotonic()
        try:
            result = agent.invoke({"messages": [("user", query)]}, config=config)
        except Exception as exc:
            self._audit.log_llm_event(
                conversation_id,
                user_id,
                provider="groq",
                model=model_name,
                success=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error_message=str(exc),
            )
            raise
        latency_ms = int((time.monotonic() - started) * 1000)
        answer = result["messages"][-1].content

        usage = self._extract_token_usage(result)
        self._audit.log_llm_event(
            conversation_id,
            user_id,
            provider="groq",
            model=model_name,
            success=True,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

        sources = self._extract_sources(answer)

        self._messages.add(conversation_id, role="user", content=query)
        self._audit.log_message_event(conversation_id, user_id, role="user", content_length=len(query))
        self._messages.add(conversation_id, role="assistant", content=answer, sources=sources)
        self._audit.log_message_event(
            conversation_id, user_id, role="assistant", content_length=len(answer)
        )
        self._conversations.touch(conversation_id)

        return answer, sources

    # ── Internals ─────────────────────────────────────────────────────────

    def _optional_vlm_tool(self, conversation_id: int, user_id: int) -> list:
        """The VLM tool depends on a locally-running Ollama instance, which
        isn't guaranteed to be available in every environment (see
        CLAUDE.md). Import lazily and skip it rather than failing chat
        entirely if VLM/imagellm.py's dependencies aren't usable. Which
        Ollama endpoint/model it's bound to comes from config.yaml, threaded
        in via Backend/deps.py -> Backend/conversations_router.py."""
        if not self._ollama_url or not self._ollama_vlm_model:
            return []
        try:
            from VLM.imagellm import build_read_image_tool

            return [
                build_read_image_tool(
                    self._ollama_url,
                    self._ollama_vlm_model,
                    audit_logger=self._audit,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
            ]
        except Exception:
            return []

    @staticmethod
    def _extract_token_usage(result: dict) -> dict:
        """Best-effort token accounting from the last AIMessage's
        usage_metadata (populated by langchain-groq) — absent for fake/test
        LLMs, in which case log_llm_event just records None for all three."""
        for message in reversed(result.get("messages", [])):
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return {
                    "prompt_tokens": usage.get("input_tokens"),
                    "completion_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
        return {}

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
