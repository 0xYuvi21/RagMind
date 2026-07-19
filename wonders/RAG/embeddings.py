"""
EmbeddingProvider implementation — wraps the HuggingFace all-MiniLM-L6-v2
(sentence-transformers) model VectorStore already defaulted to inline. Pulled
out into its own Strategy so the composition root (Backend/deps.py) can pass
an explicit, swappable embedding backend into VectorStoreFactory instead of
relying on VectorStore's internal default — swapping providers (e.g. to an
OpenAI/Groq-hosted embedding model) means writing one new class here, no
changes anywhere else.
"""

from __future__ import annotations

# Suppress the harmless HuggingFace "UNEXPECTED key: embeddings.position_ids" warning
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class HuggingFaceEmbeddingProvider:
    """EmbeddingProvider protocol implementation."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings
