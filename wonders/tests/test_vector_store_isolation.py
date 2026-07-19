"""
Proves the multi-tenancy contract: chat B can never retrieve chat A's chunks,
even via a crafted query that exactly matches chat A's content — because
each chat gets a physically separate Chroma collection (see
RAG/vector_store.py's VectorStoreFactory), not a shared collection with a
metadata filter that a caller could omit.
"""

from __future__ import annotations

from RAG.vector_store import VectorStoreFactory
from tests.fakes import make_fake_embeddings


def _factory(tmp_path) -> VectorStoreFactory:
    return VectorStoreFactory(persist_dir=str(tmp_path / "chroma"), embedding_fn=make_fake_embeddings())


def test_chats_get_distinct_collections(tmp_path):
    factory = _factory(tmp_path)
    store_a = factory.get_or_create("chat-a")
    store_b = factory.get_or_create("chat-b")
    assert store_a.collection_name() != store_b.collection_name()


def test_get_or_create_returns_same_instance_for_same_chat(tmp_path):
    factory = _factory(tmp_path)
    assert factory.get_or_create("chat-a") is factory.get_or_create("chat-a")


def test_chat_b_cannot_retrieve_chat_a_content_even_with_exact_query(tmp_path):
    factory = _factory(tmp_path)
    store_a = factory.get_or_create("chat-a")
    store_b = factory.get_or_create("chat-b")

    secret_text = "The secret launch code for Project Nightingale is ALPHA-NINE-NINE."
    store_a.add_file_stream(file_content=secret_text.encode("utf-8"), file_name="secret.txt")

    # Crafted query using the EXACT text that exists only in chat A.
    results = store_b.similarity_search(secret_text, k=5)
    assert results == []
    assert store_b.count() == 0
    assert store_a.count() > 0


def test_deleting_one_chat_does_not_affect_another(tmp_path):
    factory = _factory(tmp_path)
    store_a = factory.get_or_create("chat-a")
    store_b = factory.get_or_create("chat-b")

    store_a.add_file_stream(file_content=b"chat A document content", file_name="a.txt")
    store_b.add_file_stream(file_content=b"chat B document content", file_name="b.txt")

    factory.delete_for_chat("chat-a")

    # chat B is untouched
    assert store_b.count() > 0
    # A fresh handle to chat A is empty (collection was dropped, not just cached-cleared)
    fresh_a = factory.get_or_create("chat-a")
    assert fresh_a.count() == 0


def test_two_users_uploading_same_filename_stay_isolated(tmp_path):
    """Same filename in two different chats must not collide or leak."""
    factory = _factory(tmp_path)
    store_1 = factory.get_or_create("1")
    store_2 = factory.get_or_create("2")

    store_1.add_file_stream(file_content=b"user one notes about biology", file_name="notes.txt")
    store_2.add_file_stream(file_content=b"user two notes about chemistry", file_name="notes.txt")

    results_from_2 = store_2.similarity_search("biology", k=5)
    assert all("biology" not in r.page_content for r in results_from_2)
