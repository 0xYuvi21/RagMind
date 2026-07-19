"""
Proves the LangGraph checkpointer's short-term memory contract: a follow-up
question in the SAME conversation resolves prior context (the fake model's
input includes the earlier turn), while a DIFFERENT conversation's agent call
never sees another conversation's history — thread_id = chat_id isolation
(Retrieve/llmquery.py's build_chat_agent + Retrieve/checkpointer.py).
"""

from __future__ import annotations

from tests.fakes import make_final_answer


def _combined_text(messages) -> str:
    return " ".join(
        m.content for m in messages if isinstance(getattr(m, "content", None), str)
    )


def test_followup_in_same_conversation_sees_prior_turn(client, auth_headers, llm_script):
    llm_script["responses"] = [make_final_answer("Answer:\nParis.\n\nSources:\n\nFollow-up:\n")]

    conversation = client.post("/conversations", json={"title": "geo"}, headers=auth_headers).json()

    first = client.post(
        f"/conversations/{conversation['id']}/chat",
        json={"query": "What is the capital of France?"},
        headers=auth_headers,
    )
    assert first.status_code == 200

    second = client.post(
        f"/conversations/{conversation['id']}/chat",
        json={"query": "What is its population?"},
        headers=auth_headers,
    )
    assert second.status_code == 200

    assert len(llm_script["capture"]) == 2
    second_call_text = _combined_text(llm_script["capture"][1])
    assert "What is the capital of France?" in second_call_text
    assert "Paris" in second_call_text
    assert "What is its population?" in second_call_text


def test_different_conversation_does_not_see_other_conversations_history(
    client, auth_headers, llm_script
):
    llm_script["responses"] = [make_final_answer("Answer:\nOk.\n\nSources:\n\nFollow-up:\n")]

    conv_a = client.post("/conversations", json={"title": "a"}, headers=auth_headers).json()
    conv_b = client.post("/conversations", json={"title": "b"}, headers=auth_headers).json()

    client.post(
        f"/conversations/{conv_a['id']}/chat",
        json={"query": "Remember this: the secret word is zebra."},
        headers=auth_headers,
    )
    client.post(
        f"/conversations/{conv_b['id']}/chat",
        json={"query": "What did I just tell you to remember?"},
        headers=auth_headers,
    )

    assert len(llm_script["capture"]) == 2
    b_call_text = _combined_text(llm_script["capture"][1])
    assert "zebra" not in b_call_text


def test_message_history_is_persisted_across_turns(client, auth_headers, llm_script):
    llm_script["responses"] = [make_final_answer("Answer:\nSure.\n\nSources:\n\nFollow-up:\n")]
    conversation = client.post("/conversations", json={"title": "c"}, headers=auth_headers).json()

    client.post(
        f"/conversations/{conversation['id']}/chat",
        json={"query": "Hello there"},
        headers=auth_headers,
    )
    client.post(
        f"/conversations/{conversation['id']}/chat",
        json={"query": "Follow-up question"},
        headers=auth_headers,
    )

    messages = client.get(f"/conversations/{conversation['id']}/messages", headers=auth_headers).json()
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "Hello there"
    assert messages[2]["content"] == "Follow-up question"
