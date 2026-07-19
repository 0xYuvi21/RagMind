"""Unit tests for RAG/chunking.py's DocumentChunker."""

from __future__ import annotations

from langchain_core.documents import Document

from RAG.chunking import DocumentChunker


def _doc(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_chunk_respects_size_and_overlap():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    long_text = " ".join(f"word{i}" for i in range(200))
    chunks = chunker.chunk([_doc(long_text, source="a.txt")])

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 60  # some slack for the splitter's separator rules


def test_chunk_preserves_and_extends_metadata():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    long_text = " ".join(f"word{i}" for i in range(100))
    chunks = chunker.chunk([_doc(long_text, source="a.txt", file_name="a.txt")])

    for chunk in chunks:
        assert chunk.metadata["source"] == "a.txt"
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "chunk_total" in chunk.metadata


def test_chunk_dedups_identical_content_by_md5():
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
    # Two distinct source documents that happen to produce byte-identical chunks
    chunks = chunker.chunk([_doc("duplicate content", source="a.txt"), _doc("duplicate content", source="b.txt")])

    assert len(chunks) == 1


def test_chunk_id_is_stable_for_same_content():
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
    chunks_a = chunker.chunk([_doc("stable content here", source="a.txt")])
    chunks_b = chunker.chunk([_doc("stable content here", source="b.txt")])

    assert chunks_a[0].metadata["chunk_id"] == chunks_b[0].metadata["chunk_id"]


def test_chunk_empty_input_returns_empty_list():
    chunker = DocumentChunker()
    assert chunker.chunk([]) == []


def test_stats_on_empty_and_nonempty():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    assert chunker.stats([]) == {"total_chunks": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}

    long_text = " ".join(f"word{i}" for i in range(100))
    chunks = chunker.chunk([_doc(long_text, source="a.txt")])
    stats = chunker.stats(chunks)
    assert stats["total_chunks"] == len(chunks)
    assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]


def test_chunk_texts_convenience_method():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
    docs = chunker.chunk_texts(["hello world", "goodbye world"], metadatas=[{"a": 1}, {"a": 2}])
    assert len(docs) == 2
    assert docs[0].metadata["a"] == 1
