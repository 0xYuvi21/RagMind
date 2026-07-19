"""
CheckpointBackend implementations — LangGraph's persistence layer for
short-term conversational memory (message history), keyed by thread_id.

Default: SqliteCheckpointBackend — conversations survive a server restart,
which a "come back and continue this chat later" product needs. Test/dev
opt-out: InMemoryCheckpointBackend (CHECKPOINTER_BACKEND=memory), which is
also what the test suite uses so tests never touch disk.

thread_id = str(chat_id): each conversation's LangGraph state is naturally
isolated from every other conversation's state, the same way each
conversation's vector store is isolated (see RAG/vector_store.py's
VectorStoreFactory).
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver


class InMemoryCheckpointBackend:
    """CheckpointBackend protocol implementation backed by MemorySaver."""

    def __init__(self):
        self._saver = MemorySaver()

    def get_saver(self) -> BaseCheckpointSaver:
        return self._saver


class SqliteCheckpointBackend:
    """CheckpointBackend protocol implementation backed by LangGraph's
    SqliteSaver, persisting agent state across restarts."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        from langgraph.checkpoint.sqlite import SqliteSaver

        import sqlite3

        conn = sqlite3.connect(db_path, check_same_thread=False)
        self._saver = SqliteSaver(conn)
        self._saver.setup()

    def get_saver(self) -> BaseCheckpointSaver:
        return self._saver


def build_checkpoint_backend(backend: str, db_path: str):
    """Factory used by the composition root (Backend/deps.py)."""
    if backend == "memory":
        return InMemoryCheckpointBackend()
    if backend == "sqlite":
        return SqliteCheckpointBackend(db_path)
    raise ValueError(f"Unknown checkpointer backend: {backend!r}")
