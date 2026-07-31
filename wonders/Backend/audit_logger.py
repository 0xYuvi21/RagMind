"""
AuditLogger implementations (Backend/interfaces.py's AuditLogger Protocol).

SqlAlchemyAuditLogger is the real implementation: one INSERT per event, into
whichever audit_* table (Backend/audit_models.py) matches the event category,
committed on its own short-lived session (built from a session_factory
injected at construction — same DI pattern as every repository in
Backend/repositories.py).

NullAuditLogger is a no-op standed in whenever the audit Postgres database
isn't configured (empty credentials — see config.py's audit_database_url) or
unreachable, following the same defensive-degradation pattern already used
for the optional VLM tool (ConversationService._optional_vlm_tool) and the
Kafka producer: a missing/broken audit sink must never fail the actual user
action being audited. Both classes are drop-in substitutable for each other
(Liskov Substitution) since both satisfy the same AuditLogger Protocol.
"""

from __future__ import annotations

import logging
from typing import Optional

from Backend.audit_models import (
    AuthAuditLog,
    ConversationAuditLog,
    IngestionAuditLog,
    LLMEventAuditLog,
    MessageAuditLog,
    VLMEventAuditLog,
)

logger = logging.getLogger("audit")


class SqlAlchemyAuditLogger:
    """Writes every event to the dedicated audit Postgres database. Never
    raises — a broken audit sink degrades observability, not the feature
    being audited (mirrors ConversationService.ingest_upload's handling of a
    down Kafka broker)."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def _insert(self, row) -> None:
        session = self._session_factory()
        try:
            session.add(row)
            session.commit()
        except Exception:
            logger.exception("Failed to write audit event: %s", type(row).__name__)
        finally:
            session.close()

    def log_auth_event(
        self, action: str, email: str, success: bool, user_id: Optional[int] = None, detail: Optional[str] = None
    ) -> None:
        self._insert(
            AuthAuditLog(action=action, email=email, success=success, user_id=user_id, detail=detail)
        )

    def log_conversation_event(
        self, action: str, conversation_id: int, user_id: int, detail: Optional[str] = None
    ) -> None:
        self._insert(
            ConversationAuditLog(
                action=action, conversation_id=conversation_id, user_id=user_id, detail=detail
            )
        )

    def log_message_event(
        self, conversation_id: int, user_id: int, role: str, content_length: int
    ) -> None:
        self._insert(
            MessageAuditLog(
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content_length=content_length,
            )
        )

    def log_ingestion_event(
        self,
        conversation_id: int,
        user_id: int,
        job_id: int,
        file_name: str,
        status: str,
        chunks_added: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self._insert(
            IngestionAuditLog(
                conversation_id=conversation_id,
                user_id=user_id,
                job_id=job_id,
                file_name=file_name,
                status=status,
                chunks_added=chunks_added,
                error_message=error_message,
            )
        )

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
    ) -> None:
        self._insert(
            LLMEventAuditLog(
                conversation_id=conversation_id,
                user_id=user_id,
                provider=provider,
                model=model,
                success=success,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                error_message=error_message,
            )
        )

    def log_vlm_event(
        self,
        model: str,
        image_path: str,
        success: bool,
        conversation_id: Optional[int] = None,
        user_id: Optional[int] = None,
        latency_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self._insert(
            VLMEventAuditLog(
                conversation_id=conversation_id,
                user_id=user_id,
                model=model,
                image_path=image_path,
                success=success,
                latency_ms=latency_ms,
                error_message=error_message,
            )
        )


class NullAuditLogger:
    """No-op AuditLogger — used when audit_database_url is unset (empty
    Postgres credentials, the default until they're filled in) so the app
    runs exactly as before with no audit trail, rather than failing to
    start. Backend/deps.py picks between this and SqlAlchemyAuditLogger
    based solely on whether audit_database_url is configured."""

    def log_auth_event(self, *args, **kwargs) -> None:
        pass

    def log_conversation_event(self, *args, **kwargs) -> None:
        pass

    def log_message_event(self, *args, **kwargs) -> None:
        pass

    def log_ingestion_event(self, *args, **kwargs) -> None:
        pass

    def log_llm_event(self, *args, **kwargs) -> None:
        pass

    def log_vlm_event(self, *args, **kwargs) -> None:
        pass
