from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

logger = logging.getLogger(__name__)

from Backend.auth_dependencies import get_current_user
from Backend.config import Settings
from Backend.conversation_service import ConversationNotFoundError, ConversationService
from Backend.db_models import User
from Backend.deps import (
    get_checkpoint_backend,
    get_conversation_repository,
    get_llm_provider,
    get_message_repository,
    get_settings_cached,
    get_vector_store_factory,
)
from Backend.repositories import SqlAlchemyConversationRepository, SqlAlchemyMessageRepository
from Backend.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationRenameRequest,
    ConversationResponse,
    MessageResponse,
    UploadResponse,
)
from RAG.vector_store import VectorStoreFactory

router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_conversation_service(
    conversation_repository: SqlAlchemyConversationRepository = Depends(get_conversation_repository),
    message_repository: SqlAlchemyMessageRepository = Depends(get_message_repository),
    vector_store_factory: VectorStoreFactory = Depends(get_vector_store_factory),
    llm_provider=Depends(get_llm_provider),
    checkpoint_backend=Depends(get_checkpoint_backend),
    settings: Settings = Depends(get_settings_cached),
) -> ConversationService:
    return ConversationService(
        conversation_repository=conversation_repository,
        message_repository=message_repository,
        vector_store_factory=vector_store_factory,
        llm_provider=llm_provider,
        checkpoint_backend=checkpoint_backend,
        uploads_dir=settings.uploads_dir,
    )


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")


@router.get("", response_model=List[ConversationResponse])
def list_conversations(
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    return service.list_for_user(current_user.id)


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreateRequest,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    return service.create(current_user.id, payload.title)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def rename_conversation(
    conversation_id: int,
    payload: ConversationRenameRequest,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return service.rename(conversation_id, current_user.id, payload.title)
    except ConversationNotFoundError:
        raise _not_found()


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        service.delete(conversation_id, current_user.id)
    except ConversationNotFoundError:
        raise _not_found()


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
def list_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return service.list_messages(conversation_id, current_user.id)
    except ConversationNotFoundError:
        raise _not_found()


@router.post("/{conversation_id}/upload", response_model=UploadResponse)
async def upload_document(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
    file: UploadFile = File(...),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    try:
        chunks_added = service.ingest_upload(conversation_id, current_user.id, content, file.filename)
    except ConversationNotFoundError:
        raise _not_found()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return UploadResponse(file_name=file.filename, chunks_added=chunks_added)


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
def chat(
    conversation_id: int,
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        answer, sources = service.ask(conversation_id, current_user.id, payload.query)
    except ConversationNotFoundError:
        raise _not_found()
    except Exception:  # graceful degradation on LLM/tool failure
        logger.exception("Chat agent invocation failed for conversation %s", conversation_id)
        return ChatResponse(
            text=(
                "Sorry, I encountered an error while processing your request. "
                "Please try again or rephrase your query."
            ),
            sources=[],
        )

    return ChatResponse(text=answer, sources=sources)
