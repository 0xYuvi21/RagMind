from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from Backend.auth_service import AuthService, EmailAlreadyRegisteredError, InvalidCredentialsError
from Backend.config import Settings
from Backend.deps import get_audit_logger, get_settings_cached, get_user_repository
from Backend.interfaces import AuditLogger
from Backend.repositories import SqlAlchemyUserRepository
from Backend.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(
    user_repository: SqlAlchemyUserRepository = Depends(get_user_repository),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> AuthService:
    return AuthService(user_repository, audit_logger)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, auth_service: AuthService = Depends(get_auth_service)):
    try:
        user = auth_service.register(payload.email, payload.password)
    except EmailAlreadyRegisteredError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return user


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings_cached),
):
    try:
        user = auth_service.authenticate(payload.email, payload.password)
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password"
        )

    token = auth_service.issue_token(
        user, settings.jwt_secret_key, settings.jwt_algorithm, settings.jwt_expire_minutes
    )
    return TokenResponse(access_token=token)
