"""
vector_store.py
===============
Embedding pipeline:
    ingest.py (DocumentIngestor)
        → chunking.py (DocumentChunker)
            → ChromaDB (HuggingFace embeddings)

The VectorStore class wires those three steps together.

Quick start
-----------
from RAG.vector_store import VectorStore

store = VectorStore()

# Ingest a file — auto-ingests, chunks, then embeds and stores
store.add_file("path/to/paper.pdf")

# Or ingest a whole folder
store.add_directory("path/to/docs/", recursive=True)

# Query
results = store.similarity_search("What is RAG?", k=5)

# Use as a LangChain retriever in a chain
retriever = store.as_retriever(k=4)
"""

from __future__ import annotations

# Suppress the harmless HuggingFace "UNEXPECTED key: embeddings.position_ids" warning
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Pipeline imports — both modules live in the same RAG/ package
from RAG.ingest import DocumentIngestor
from RAG.chunking import DocumentChunker
from RAG.ocr import OcrProvider

# ──────────────────────────────────────────────
# Defaults — tweak here or via constructor args
# ──────────────────────────────────────────────
DEFAULT_PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
DEFAULT_COLLECTION = "rag_collection"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100


class VectorStore:
    """
    Wires the full RAG ingestion pipeline:
        DocumentIngestor → DocumentChunker → ChromaDB

    Parameters
    ----------
    persist_dir   : str       – on-disk path for ChromaDB data.
    collection    : str       – name of the Chroma collection.
    embedding_fn  : Embeddings – swap in any LangChain embedding model.
                    Defaults to HuggingFace all-MiniLM-L6-v2 (local, no API key).
    chunk_size    : int       – characters per chunk.
    chunk_overlap : int       – overlapping characters between adjacent chunks.
    ocr_provider  : OcrProvider – optional strategy for extracting literal
                    text out of images embedded in PDF/DOCX uploads (see
                    RAG/ocr.py). None by default (no OCR run).
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection: str = DEFAULT_COLLECTION,
        embedding_fn: Optional[Embeddings] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        ocr_provider: Optional[OcrProvider] = None,
    ):
        self._persist_dir = persist_dir
        self._collection = collection

        # ── Pipeline components ──────────────────────────────────────────
        self._ingestor = DocumentIngestor(ocr_provider=ocr_provider)
        self._chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # ── Embedding model ──────────────────────────────────────────────
        self._embeddings: Embeddings = embedding_fn or HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── ChromaDB ─────────────────────────────────────────────────────
        os.makedirs(persist_dir, exist_ok=True)

        self._db: Chroma = Chroma(
            collection_name=collection,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

        # Store is ready — output handled by caller

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion — ingest.py → chunking.py → store
    # ──────────────────────────────────────────────────────────────────────

    def add_file(
        self,
        file_path: str,
        source_tag: Optional[str] = None,
        image_output_dir: Optional[str] = None,
    ) -> int:
        """
        Full pipeline for a single file:
            DocumentIngestor.ingest()
                → DocumentChunker.chunk()
                    → embed & store in ChromaDB

        Parameters
        ----------
        file_path  : str – path to a .pdf, .txt, etc.
        source_tag : str – optional metadata label for filtering later.
        image_output_dir : str – where embedded images get extracted to
                           (see DocumentIngestor); defaults to the file's own
                           directory if omitted.

        Returns
        -------
        int – number of chunks stored.
        """
        docs = self._ingestor.ingest(file_path, image_output_dir=image_output_dir)  # ← ingest.py
        chunks = self._chunker.chunk(docs)  # ← chunking.py

        if source_tag:
            for c in chunks:
                c.metadata["source_tag"] = source_tag

        return self._upsert(chunks)

    def add_file_stream(
        self,
        file_content: bytes,
        file_name: str,
        source_tag: Optional[str] = None,
        image_output_dir: Optional[str] = None,
    ) -> int:
        """
        Full pipeline for a dynamically uploaded file (in memory):
            DocumentIngestor.ingest_file_stream()
                → DocumentChunker.chunk()
                    → embed & store in ChromaDB

        Parameters
        ----------
        file_content : bytes – raw bytes of the file.
        file_name    : str   – name of the file (e.g. "report.pdf").
        source_tag   : str   – optional metadata label for filtering later.
        image_output_dir : str – where embedded images get extracted to
                           (see DocumentIngestor); the caller should pass a
                           persistent directory (e.g. this conversation's
                           upload folder) since the uploaded file itself is
                           only held in a temp file during parsing.

        Returns
        -------
        int – number of chunks stored.
        """
        docs = self._ingestor.ingest_file_stream(
            file_content, file_name, image_output_dir=image_output_dir
        )
        chunks = self._chunker.chunk(docs)

        if source_tag:
            for c in chunks:
                c.metadata["source_tag"] = source_tag

        return self._upsert(chunks)

    def add_directory(
        self,
        dir_path: str,
        recursive: bool = False,
        source_tag: Optional[str] = None,
    ) -> int:
        """
        Full pipeline for every supported file in a directory:
            DocumentIngestor.ingest_directory()
                → DocumentChunker.chunk()
                    → embed & store in ChromaDB

        Parameters
        ----------
        dir_path   : str  – folder to scan.
        recursive  : bool – descend into sub-folders.
        source_tag : str  – optional metadata label.

        Returns
        -------
        int – total chunks stored across all files.
        """
        docs = self._ingestor.ingest_directory(dir_path, recursive=recursive)
        chunks = self._chunker.chunk(docs)

        if source_tag:
            for c in chunks:
                c.metadata["source_tag"] = source_tag

        return self._upsert(chunks)

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Return the *k* most relevant chunks for *query*.

        Parameters
        ----------
        query  : str  – natural-language question or phrase.
        k      : int  – number of results (default 5).
        filter : dict – Chroma metadata filter, e.g. ``{"source_tag": "paper_A"}``.
        """
        return self._db.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        """
        Same as ``similarity_search`` but also returns cosine-distance scores.
        Lower score = more similar.
        """
        return self._db.similarity_search_with_score(query, k=k, filter=filter)

    def as_retriever(self, k: int = 5, **kwargs):
        """
        Return a LangChain ``VectorStoreRetriever`` for use in chains/agents.

        Parameters
        ----------
        k : int – number of docs to fetch per query.
        """
        return self._db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, **kwargs},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Management
    # ──────────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of embeddings currently in the collection."""
        return self._db._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection (keeps the DB on disk)."""
        self._db.delete_collection()
        self._db = Chroma(
            collection_name=self._collection,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )
        pass  # collection cleared

    def drop(self) -> None:
        """Permanently remove this collection from disk (no empty collection left
        behind, unlike clear()). Used when an entire conversation is deleted."""
        self._db.delete_collection()

    def collection_name(self) -> str:
        """Active Chroma collection name."""
        return self._collection

    def persist_dir(self) -> str:
        """On-disk ChromaDB directory path."""
        return self._persist_dir

    # ──────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────

    def _upsert(self, chunks: List[Document]) -> int:
        """
        Embed *chunks* and add them to Chroma.
        Uses ``chunk_id`` from DocumentChunker as the Chroma document ID
        so re-ingesting the same file never creates duplicate vectors.
        """
        if not chunks:
            return 0

        # Chroma doesn't support complex metadata types (like dicts for bounding boxes)
        clean_chunks = filter_complex_metadata(chunks)

        ids = [
            c.metadata.get("chunk_id", f"chunk_{i}") for i, c in enumerate(clean_chunks)
        ]

        self._db.add_documents(clean_chunks, ids=ids)
        return len(clean_chunks)


# ──────────────────────────────────────────────────────────────────────────
# Multi-tenancy: one Chroma COLLECTION PER CONVERSATION
# ──────────────────────────────────────────────────────────────────────────
#
# Isolation strategy: rather than one shared collection filtered by a chat_id
# metadata field (a filter some future route/tool could forget to apply),
# every conversation gets its own physically separate Chroma collection.
# VectorStore already accepts an arbitrary `collection` name and shares one
# on-disk `persist_dir`, so per-chat isolation needs no change to VectorStore
# itself — just a factory that mints/caches one instance per chat_id with a
# deterministic collection name. See SYSTEM_DESIGN.md for the trade-off.
class VectorStoreFactory:
    """Resolves/creates the isolated VectorStore for a given conversation id."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_fn: Optional[Embeddings] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        ocr_provider: Optional[OcrProvider] = None,
    ):
        self._persist_dir = persist_dir
        self._embedding_fn = embedding_fn
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._ocr_provider = ocr_provider
        self._cache: dict[str, "VectorStore"] = {}

    @staticmethod
    def collection_name_for(chat_id: str) -> str:
        return f"chat_{chat_id}"

    def get_or_create(self, chat_id: str) -> "VectorStore":
        collection = self.collection_name_for(chat_id)
        if collection not in self._cache:
            self._cache[collection] = VectorStore(
                persist_dir=self._persist_dir,
                collection=collection,
                embedding_fn=self._embedding_fn,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                ocr_provider=self._ocr_provider,
            )
        return self._cache[collection]

    def delete_for_chat(self, chat_id: str) -> None:
        collection = self.collection_name_for(chat_id)
        store = self._cache.pop(collection, None)
        if store is not None:
            store.drop()


# ──────────────────────────────────────────────────────────────────────────
# Quick demo / manual test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_PDF = "C:/Users/muthi/Desktop/wonders/RAG/docs/CHATBOT_Architecture_Design_and_Developm.pdf"

    # 1. Build store (opens existing DB or creates it fresh)
    store = VectorStore(
        persist_dir=DEFAULT_PERSIST_DIR,
        collection=DEFAULT_COLLECTION,
        chunk_size=512,
        chunk_overlap=100,
    )

    # 2. ingest.py → chunking.py → ChromaDB
    added = store.add_file(SAMPLE_PDF, source_tag="chatbot_paper")
    print(f"\nChunks added      : {added}")
    print(f"Total in store    : {store.count()}")

    # 3. Query
    query = "Explain Malicious Chatbots?"
    print(f"\nTop-3 results for: {query!r}\n")

    for i, (doc, score) in enumerate(store.similarity_search_with_score(query, k=3), 1):
        m = doc.metadata
        print(
            f"  [{i}] score={score:.4f}  "
            f"{m.get('research_paper_name', '')}  "
            f"chunk {m.get('chunk_index', '?')}/{m.get('chunk_total', '?')}"
        )
        print(f"       {doc.page_content.strip()!r}")
        print()
