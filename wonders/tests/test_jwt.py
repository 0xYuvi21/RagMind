"""Unit tests for Backend/security.py's JWT helpers."""

from __future__ import annotations

import datetime
import time

import jwt as pyjwt
import pytest

from Backend.security import TokenError, create_access_token, decode_access_token

SECRET = "test-secret-key-at-least-32-bytes-long"


def test_encode_decode_roundtrip():
    token = create_access_token(subject="42", secret_key=SECRET, expires_minutes=60)
    payload = decode_access_token(token, secret_key=SECRET)
    assert payload["sub"] == "42"
    assert "exp" in payload
    assert "iat" in payload


def test_expired_token_is_rejected():
    token = create_access_token(subject="1", secret_key=SECRET, expires_minutes=-1)
    with pytest.raises(TokenError):
        decode_access_token(token, secret_key=SECRET)


def test_tampered_signature_is_rejected():
    token = create_access_token(subject="1", secret_key=SECRET, expires_minutes=60)
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(TokenError):
        decode_access_token(tampered, secret_key=SECRET)


def test_wrong_secret_is_rejected():
    token = create_access_token(subject="1", secret_key=SECRET, expires_minutes=60)
    with pytest.raises(TokenError):
        decode_access_token(token, secret_key="a-completely-different-secret-key")


def test_wrong_algorithm_is_rejected():
    token = create_access_token(subject="1", secret_key=SECRET, algorithm="HS256", expires_minutes=60)
    with pytest.raises(TokenError):
        decode_access_token(token, secret_key=SECRET, algorithm="HS384")


def test_malformed_token_is_rejected():
    with pytest.raises(TokenError):
        decode_access_token("not-a-jwt-at-all", secret_key=SECRET)
