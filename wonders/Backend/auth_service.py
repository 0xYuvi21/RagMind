"""Registration/login business logic — kept separate from the router so it's
testable without spinning up FastAPI's request/response cycle."""

from __future__ import annotations

from Backend.audit_logger import NullAuditLogger
from Backend.db_models import User
from Backend.interfaces import AuditLogger
from Backend.repositories import SqlAlchemyUserRepository
from Backend.security import create_access_token, hash_password, verify_password


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class AuthService:
    def __init__(
        self,
        user_repository: SqlAlchemyUserRepository,
        audit_logger: AuditLogger | None = None,
    ):
        self._user_repository = user_repository
        self._audit = audit_logger or NullAuditLogger()

    def register(self, email: str, password: str) -> User:
        if self._user_repository.get_by_email(email) is not None:
            self._audit.log_auth_event(
                "register", email, success=False, detail="Email already registered"
            )
            raise EmailAlreadyRegisteredError(email)
        user = self._user_repository.create(email=email, hashed_password=hash_password(password))
        self._audit.log_auth_event("register", email, success=True, user_id=user.id)
        return user

    def authenticate(self, email: str, password: str) -> User:
        user = self._user_repository.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            self._audit.log_auth_event(
                "login", email, success=False, detail="Invalid credentials"
            )
            raise InvalidCredentialsError()
        self._audit.log_auth_event("login", email, success=True, user_id=user.id)
        return user

    @staticmethod
    def issue_token(user: User, secret_key: str, algorithm: str, expires_minutes: int) -> str:
        return create_access_token(
            subject=str(user.id),
            secret_key=secret_key,
            algorithm=algorithm,
            expires_minutes=expires_minutes,
        )
