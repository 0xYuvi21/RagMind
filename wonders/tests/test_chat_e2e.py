"""
End-to-end: upload a file into a conversation, ask about it in the same
session (no server restart), and confirm the answer is grounded in that
upload — plus confirm a second conversation for the SAME user cannot see the
first conversation's uploaded documents (isolation enforced through the full
HTTP stack, not just at the VectorStore layer — see
tests/test_vector_store_isolation.py for that lower-level proof).
"""

from __future__ import annotations

from tests.conftest import register_and_login
from tests.fakes import make_final_answer, make_tool_calling_response


def test_upload_then_ask_in_same_session_returns_grounded_answer(client, auth_headers, llm_script):
    conversation = client.post("/conversations", json={"title": "notes"}, headers=auth_headers).json()

    content = b"RagMind uses ChromaDB for vector storage and Groq for the language model."
    upload = client.post(
        f"/conversations/{conversation['id']}/upload",
        headers=auth_headers,
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert upload.status_code == 202
    assert upload.json()["status"] == "done"  # FakeIngestionProducer runs inline, no real queue hop
    assert upload.json()["chunks_added"] >= 1

    llm_script["responses"] = [
        make_tool_calling_response("search_knowledge_base", {"query": "what vector store does RagMind use"}),
        make_final_answer(
            "Answer:\nRagMind uses ChromaDB.\n\nSources:\n- notes.txt\n\nFollow-up:\nWhat embedding model?"
        ),
    ]

    chat = client.post(
        f"/conversations/{conversation['id']}/chat",
        json={"query": "What vector store does RagMind use?"},
        headers=auth_headers,
    )
    assert chat.status_code == 200
    body = chat.json()
    assert "ChromaDB" in body["text"]
    assert "notes.txt" in body["sources"]


def test_second_conversation_cannot_see_first_conversations_documents(
    client, auth_headers, llm_script
):
    conv1 = client.post("/conversations", json={"title": "c1"}, headers=auth_headers).json()
    conv2 = client.post("/conversations", json={"title": "c2"}, headers=auth_headers).json()

    secret = b"The hidden treasure is buried under the old oak tree behind the library."
    client.post(
        f"/conversations/{conv1['id']}/upload",
        headers=auth_headers,
        files={"file": ("treasure.txt", secret, "text/plain")},
    )

    llm_script["responses"] = [
        make_tool_calling_response("search_knowledge_base", {"query": "hidden treasure oak tree"}),
        make_final_answer(
            "Answer:\nI could not find anything relevant in this conversation.\n\nSources:\n\nFollow-up:\n"
        ),
    ]
    chat = client.post(
        f"/conversations/{conv2['id']}/chat",
        json={"query": "Where is the hidden treasure?"},
        headers=auth_headers,
    )
    assert chat.status_code == 200

    # The tool result fed back to the model (2nd model call's input) must not
    # contain conv1's secret content — proves retrieval itself found nothing,
    # not just that the scripted answer happens to omit it.
    second_call_messages = llm_script["capture"][-1]
    combined = " ".join(
        m.content for m in second_call_messages if isinstance(getattr(m, "content", None), str)
    )
    assert "oak tree" not in combined


def test_upload_status_endpoint_reflects_completed_job(client, auth_headers):
    conversation = client.post("/conversations", json={"title": "notes"}, headers=auth_headers).json()

    content = b"A short note about mitochondria."
    upload = client.post(
        f"/conversations/{conversation['id']}/upload",
        headers=auth_headers,
        files={"file": ("bio.txt", content, "text/plain")},
    )
    job_id = upload.json()["id"]

    status_response = client.get(
        f"/conversations/{conversation['id']}/uploads/{job_id}", headers=auth_headers
    )
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "done"
    assert body["chunks_added"] >= 1
    assert body["file_name"] == "bio.txt"


def test_upload_status_unknown_job_returns_404(client, auth_headers):
    conversation = client.post("/conversations", json={"title": "notes"}, headers=auth_headers).json()
    response = client.get(
        f"/conversations/{conversation['id']}/uploads/999999", headers=auth_headers
    )
    assert response.status_code == 404


def test_upload_status_hidden_from_other_users(client):
    headers_a = register_and_login(client, "uploader@example.com")
    headers_b = register_and_login(client, "snoop@example.com")

    conversation = client.post("/conversations", json={"title": "notes"}, headers=headers_a).json()
    upload = client.post(
        f"/conversations/{conversation['id']}/upload",
        headers=headers_a,
        files={"file": ("bio.txt", b"secret notes", "text/plain")},
    )
    job_id = upload.json()["id"]

    response = client.get(
        f"/conversations/{conversation['id']}/uploads/{job_id}", headers=headers_b
    )
    assert response.status_code == 404


def test_unsupported_extension_rejected_synchronously(client, auth_headers):
    """Extension validation happens before enqueueing, so a bad file type is
    a fast 400 at upload time, not a job that shows up "failed" later."""
    conversation = client.post("/conversations", json={"title": "notes"}, headers=auth_headers).json()
    response = client.post(
        f"/conversations/{conversation['id']}/upload",
        headers=auth_headers,
        files={"file": ("virus.exe", b"not a real document", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_to_nonexistent_conversation_returns_404(client, auth_headers):
    response = client.post(
        "/conversations/999999/upload",
        headers=auth_headers,
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 404


def test_chat_with_missing_groq_key_degrades_gracefully(client, auth_headers, llm_script):
    """When the scripted/real LLM call fails outright, the chat endpoint
    should return a friendly message + 200, not a 500."""

    class ExplodingProvider:
        def get_chat_model(self):
            raise EnvironmentError("boom")

    from Backend import deps
    from Backend.backend import app

    conversation = client.post("/conversations", json={"title": "x"}, headers=auth_headers).json()
    app.dependency_overrides[deps.get_llm_provider] = lambda: ExplodingProvider()

    response = client.post(
        f"/conversations/{conversation['id']}/chat",
        json={"query": "Hello?"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert "error" in response.json()["text"].lower()
