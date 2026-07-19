"""
Password hashing and JWT helpers.

Hashing uses bcrypt directly (not passlib — passlib's last release predates
bcrypt 4.x and emits a spurious "error reading bcrypt version" warning on
modern bcrypt; calling bcrypt directly avoids the dependency entirely, and
bcrypt is already a transitive dependency of this project).

A single access token (no refresh token) is issued at login, carrying the
user id as `sub`. See SYSTEM_DESIGN.md for why refresh tokens were skipped.
"""

from __future__ import annotations

import datetime
from typing import Any

import bcrypt
import jwt as pyjwt

_MAX_BCRYPT_BYTES = 72  # bcrypt silently truncates beyond this; we reject instead


class TokenError(Exception):
    """Raised for any invalid, expired, or tampered token."""


def hash_password(plain_password: str) -> str:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > _MAX_BCRYPT_BYTES:
        raise ValueError("Password is too long")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(
    subject: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 60 * 24,
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=expires_minutes),
    }
    return pyjwt.encode(payload, secret_key, algorithm=algorithm)


def decode_access_token(token: str, secret_key: str, algorithm: str = "HS256") -> dict:
    try:
        return pyjwt.decode(token, secret_key, algorithms=[algorithm])
    except pyjwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except pyjwt.InvalidTokenError as exc:
        raise TokenError("Invalid token") from exc
