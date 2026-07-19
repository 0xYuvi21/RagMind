"""End-to-end tests for /auth/register and /auth/login via FastAPI TestClient."""

from __future__ import annotations


def test_register_creates_user(client):
    response = client.post("/auth/register", json={"email": "new@example.com", "password": "password123"})
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com"
    assert "id" in body
    assert "password" not in body
    assert "hashed_password" not in body


def test_register_duplicate_email_is_rejected(client):
    client.post("/auth/register", json={"email": "dup@example.com", "password": "password123"})
    response = client.post("/auth/register", json={"email": "dup@example.com", "password": "password123"})
    assert response.status_code == 400


def test_login_with_correct_password_returns_token(client):
    client.post("/auth/register", json={"email": "ok@example.com", "password": "password123"})
    response = client.post("/auth/login", json={"email": "ok@example.com", "password": "password123"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert len(body["access_token"]) > 20


def test_login_with_wrong_password_is_rejected(client):
    client.post("/auth/register", json={"email": "ok2@example.com", "password": "password123"})
    response = client.post("/auth/login", json={"email": "ok2@example.com", "password": "wrong-password"})
    assert response.status_code == 401


def test_login_with_unknown_email_is_rejected(client):
    response = client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"})
    assert response.status_code == 401


def test_protected_route_without_token_is_rejected(client):
    response = client.get("/conversations")
    assert response.status_code == 401


def test_protected_route_with_garbage_token_is_rejected(client):
    response = client.get("/conversations", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
