"""
Repository pattern: SQLAlchemy-backed implementations of the
UserRepository/ConversationRepository/MessageRepository protocols declared in
Backend/interfaces.py.

Ownership enforcement (does conversation X belong to user Y?) deliberately
does NOT live here — these repositories only know about ids. The service
layer (Backend/conversation_service.py) checks ownership before calling
into these methods, so a mismatch can be reported as 404 without this layer
needing to know about "the current user" at all.
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from Backend.db_models import Conversation, IngestionJob, IngestionJobStatus, Message, User


class SqlAlchemyUserRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(self, email: str, hashed_password: str) -> User:
        user = User(email=email, hashed_password=hashed_password)
        self._session.add(user)
        self._session.commit()
        self._session.refresh(user)
        return user

    def get_by_email(self, email: str) -> Optional[User]:
        return self._session.query(User).filter(User.email == email).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        return self._session.get(User, user_id)


class SqlAlchemyConversationRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(self, user_id: int, title: str) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        self._session.add(conversation)
        self._session.commit()
        self._session.refresh(conversation)
        return conversation

    def list_for_user(self, user_id: int) -> List[Conversation]:
        return (
            self._session.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def get(self, conversation_id: int) -> Optional[Conversation]:
        return self._session.get(Conversation, conversation_id)

    def update_title(self, conversation_id: int, title: str) -> Optional[Conversation]:
        conversation = self.get(conversation_id)
        if conversation is None:
            return None
        conversation.title = title
        conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self._session.commit()
        self._session.refresh(conversation)
        return conversation

    def touch(self, conversation_id: int) -> None:
        conversation = self.get(conversation_id)
        if conversation is None:
            return
        conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self._session.commit()

    def delete(self, conversation_id: int) -> None:
        conversation = self.get(conversation_id)
        if conversation is None:
            return
        self._session.delete(conversation)
        self._session.commit()


class SqlAlchemyIngestionJobRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(self, conversation_id: int, file_name: str, stored_path: str) -> IngestionJob:
        job = IngestionJob(
            conversation_id=conversation_id,
            file_name=file_name,
            stored_path=stored_path,
            status=IngestionJobStatus.QUEUED,
        )
        self._session.add(job)
        self._session.commit()
        self._session.refresh(job)
        return job

    def get(self, job_id: int) -> Optional[IngestionJob]:
        return self._session.get(IngestionJob, job_id)

    def mark_processing(self, job_id: int) -> None:
        self._set_status(job_id, IngestionJobStatus.PROCESSING)

    def mark_done(self, job_id: int, chunks_added: int) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status = IngestionJobStatus.DONE
        job.chunks_added = chunks_added
        job.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self._session.commit()

    def mark_failed(self, job_id: int, error_message: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status = IngestionJobStatus.FAILED
        job.error_message = error_message
        job.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self._session.commit()

    def _set_status(self, job_id: int, status_value: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status = status_value
        job.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self._session.commit()


class SqlAlchemyMessageRepository:
    def __init__(self, session: Session):
        self._session = session

    def add(
        self, conversation_id: int, role: str, content: str, sources: Optional[List[str]] = None
    ) -> Message:
        message = Message(
            conversation_id=conversation_id, role=role, content=content, sources=sources or []
        )
        self._session.add(message)
        self._session.commit()
        self._session.refresh(message)
        return message

    def list_for_conversation(self, conversation_id: int) -> List[Message]:
        return (
            self._session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id)
            .all()
        )
