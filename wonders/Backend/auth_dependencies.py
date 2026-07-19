"""
FastAPI dependency that resolves the authenticated User from a Bearer JWT.

Every protected route depends on `get_current_user` (directly or via a
conversation-ownership dependency in conversations_router.py) instead of
trusting a client-supplied user id — this is what makes cross-user access
return 401/404 instead of silently operating on someone else's data.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from Backend.config import Settings
from Backend.db_models import User
from Backend.deps import get_settings, get_user_repository
from Backend.repositories import SqlAlchemyUserRepository
from Backend.security import TokenError, decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
    user_repository: SqlAlchemyUserRepository = Depends(get_user_repository),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token, settings.jwt_secret_key, settings.jwt_algorithm)
    except TokenError:
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = user_repository.get_by_id(int(user_id))
    if user is None:
        raise credentials_exception

    return user
