"""
Conversation CRUD + ownership enforcement. A conversation belonging to user A
must be invisible/unmodifiable by user B — checked with a 404, not a 403, so
the response doesn't confirm the id even exists (see
Backend/conversation_service.py's ConversationNotFoundError).
"""

from __future__ import annotations

from tests.conftest import register_and_login


def test_create_and_list_conversation(client, auth_headers):
    response = client.post("/conversations", json={"title": "Biology notes"}, headers=auth_headers)
    assert response.status_code == 201
    conversation_id = response.json()["id"]
    assert response.json()["title"] == "Biology notes"

    listing = client.get("/conversations", headers=auth_headers)
    assert listing.status_code == 200
    ids = [c["id"] for c in listing.json()]
    assert conversation_id in ids


def test_create_conversation_default_title(client, auth_headers):
    response = client.post("/conversations", json={}, headers=auth_headers)
    assert response.status_code == 201
    assert response.json()["title"] == "New Conversation"


def test_rename_conversation(client, auth_headers):
    created = client.post("/conversations", json={"title": "Old"}, headers=auth_headers).json()
    response = client.patch(
        f"/conversations/{created['id']}", json={"title": "New Title"}, headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["title"] == "New Title"


def test_delete_conversation(client, auth_headers):
    created = client.post("/conversations", json={"title": "To delete"}, headers=auth_headers).json()
    response = client.delete(f"/conversations/{created['id']}", headers=auth_headers)
    assert response.status_code == 204

    listing = client.get("/conversations", headers=auth_headers).json()
    assert created["id"] not in [c["id"] for c in listing]


def test_list_messages_empty_for_new_conversation(client, auth_headers):
    created = client.post("/conversations", json={}, headers=auth_headers).json()
    response = client.get(f"/conversations/{created['id']}/messages", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_user_b_cannot_see_user_a_conversation(client):
    headers_a = register_and_login(client, "owner@example.com")
    headers_b = register_and_login(client, "intruder@example.com")

    created = client.post("/conversations", json={"title": "Private"}, headers=headers_a).json()
    conversation_id = created["id"]

    assert client.get(f"/conversations/{conversation_id}/messages", headers=headers_b).status_code == 404
    assert (
        client.patch(
            f"/conversations/{conversation_id}", json={"title": "Hijacked"}, headers=headers_b
        ).status_code
        == 404
    )
    assert client.delete(f"/conversations/{conversation_id}", headers=headers_b).status_code == 404

    # And user A's conversation is untouched by B's failed attempts
    still_there = client.get("/conversations", headers=headers_a).json()
    assert any(c["id"] == conversation_id and c["title"] == "Private" for c in still_there)


def test_user_b_conversation_list_never_includes_user_a_conversations(client):
    headers_a = register_and_login(client, "alice2@example.com")
    headers_b = register_and_login(client, "bob2@example.com")

    client.post("/conversations", json={"title": "Alice's chat"}, headers=headers_a)
    response = client.get("/conversations", headers=headers_b)
    assert response.json() == []


def test_nonexistent_conversation_returns_404(client, auth_headers):
    response = client.get("/conversations/999999/messages", headers=auth_headers)
    assert response.status_code == 404
