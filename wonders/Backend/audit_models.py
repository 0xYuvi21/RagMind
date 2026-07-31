"""
Audit log ORM models — one table per action category, per explicit user
request, instead of one polymorphic `events` table. Each table's columns are
specific to what that action actually needs to record (e.g. an LLM call
needs model/token counts; a login attempt needs an email and a success
flag), which a single shared "details JSON blob" table would blur.

Every table shares the same identity/timestamp/actor shape via the
`AuditLogMixin` — this is the OOP "common base, specialized subclasses"
structure: adding a new action category later means one small model class
inheriting the mixin, not a schema migration to a monolithic table.

None of these rows are ever updated or deleted by the app — audit logs are
append-only by construction (see Backend/audit_logger.py: every method is a
single INSERT, no update()/delete() exposed).
"""

from __future__ import annotations

import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from Backend.audit_db import AuditBase


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class AuditLogMixin:
    """Common columns for every audit table: surrogate key, who did it (may
    be null for a failed login where the email doesn't resolve to a known
    user), and when. `user_id` intentionally has no ForeignKey to the app's
    `users` table — the audit database is a separate Postgres instance from
    the app's own DATABASE_URL (see Backend/audit_db.py), so a cross-database
    FK isn't possible; it's a plain denormalized reference instead."""

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    occurred_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


class AuthAuditLog(AuditBase, AuditLogMixin):
    """One row per register/login attempt — including failed logins
    (user_id null, email + success=False), which is exactly the kind of
    event a plain 200/404 response body doesn't preserve anywhere else."""

    __tablename__ = "audit_auth_events"

    action: Mapped[str] = mapped_column(String(50), nullable=False)  # "register" | "login"
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConversationAuditLog(AuditBase, AuditLogMixin):
    """One row per conversation lifecycle action (create/rename/delete)."""

    __tablename__ = "audit_conversation_events"

    action: Mapped[str] = mapped_column(String(50), nullable=False)  # "create"|"rename"|"delete"
    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class MessageAuditLog(AuditBase, AuditLogMixin):
    """One row per chat turn (a user question being asked), separate from
    the LLM-call table below since "a question was asked" and "the model
    was invoked N times to answer it" are different events (an agent may
    call tools/loop internally)."""

    __tablename__ = "audit_message_events"

    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content_length: Mapped[int] = mapped_column(Integer, nullable=False)


class IngestionAuditLog(AuditBase, AuditLogMixin):
    """One row per ingestion job state transition (queued/done/failed) —
    complements Backend/db_models.py's IngestionJob (current state) with an
    append-only history of how it got there."""

    __tablename__ = "audit_ingestion_events"

    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    job_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    chunks_added: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class LLMEventAuditLog(AuditBase, AuditLogMixin):
    """One row per LLM (Groq) invocation — explicitly called out by name in
    the request this was built from ("especially the llm events need to be
    stored"). Records enough to reconstruct cost/usage after the fact
    without storing the full prompt/response body (kept out to avoid
    duplicating the `messages` table's document content and any PII within
    it in a second, separately-secured database)."""

    __tablename__ = "audit_llm_events"

    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # "groq"
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class VLMEventAuditLog(AuditBase, AuditLogMixin):
    """One row per `read_image` tool invocation (Ollama vision model) — kept
    separate from LLMEventAuditLog since it's a different provider/model
    axis entirely (local Ollama vs. Groq) with different fields (image path,
    no token accounting from Ollama's /api/generate response)."""

    __tablename__ = "audit_vlm_events"

    conversation_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
