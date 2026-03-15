import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# Sensible defaults — tweak here or via constructor
# ─────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class DocumentChunker:
    """
    Splits a list of LangChain Documents into smaller, overlapping chunks
    using RecursiveCharacterTextSplitter.

    Each output chunk inherits the source document's metadata and gets
    two extra fields added:
        - ``chunk_index`` : 1-based position of this chunk within its source doc
        - ``chunk_total`` : total number of chunks produced from that source doc

    Usage
    -----
    from ingest import DocumentIngestor
    from chunking import DocumentChunker

    docs   = DocumentIngestor().ingest("paper.pdf")
    chunks = DocumentChunker().chunk(docs)
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
        length_function=len,
    ):
        """
        Parameters
        ----------
        chunk_size    : int  – max characters per chunk (default 512)
        chunk_overlap : int  – characters shared between adjacent chunks (default 64)
        separators    : list – ordered split boundaries; falls back to character split
        length_function : callable – how to measure chunk length (default: len)
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
            length_function=length_function,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents.

        Parameters
        ----------
        documents : List[Document]
            Output from DocumentIngestor.ingest() or .ingest_directory()

        Returns
        -------
        List[Document]
            Flat list of chunk Documents with enriched metadata.
        """
        all_chunks: List[Document] = []
        seen: set[str] = set()  # MD5 hashes of already-added chunks

        for doc in documents:
            splits = self._splitter.split_documents([doc])
            total = len(splits)

            for idx, chunk in enumerate(splits, start=1):
                chunk_id = self._md5(chunk.page_content, idx)

                if chunk_id in seen:
                    continue  # duplicate — skip silently
                seen.add(chunk_id)

                # Preserve ALL original metadata; add dedup id + position info
                chunk.metadata = {
                    **chunk.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_total": total,
                }
                all_chunks.append(chunk)

        return all_chunks

    def chunk_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Convenience method to chunk raw strings instead of Document objects.

        Parameters
        ----------
        texts     : List[str]  – raw text strings to chunk
        metadatas : List[dict] – optional per-text metadata dicts

        Returns
        -------
        List[Document]
        """
        return self._splitter.create_documents(
            texts, metadatas=metadatas or [{}] * len(texts)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _md5(text: str, idx : int) -> str:
        """Return the MD5 hex-digest of *text* (UTF-8 encoded)."""
        hasher = hashlib.md5(text.encode("utf-8")).hexdigest() 
        return hasher

    # ------------------------------------------------------------------

    def stats(self, chunks: List[Document]) -> dict:
        """
        Return basic statistics about a chunked list.

        Parameters
        ----------
        chunks : List[Document]

        Returns
        -------
        dict with keys: total_chunks, avg_chars, min_chars, max_chars
        """
        if not chunks:
            return {"total_chunks": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}

        sizes = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chars": round(sum(sizes) / len(sizes)),
            "min_chars": min(sizes),
            "max_chars": max(sizes),
        }


# ─────────────────────────────────────────────
# Quick demo / manual test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import DocumentIngestor

    sample_pdf = "C:/Users/muthi/Desktop/wonders/RAG/docs/CHATBOT_Architecture_Design_and_Developm.pdf"

    ingestor = DocumentIngestor()
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)

    docs = ingestor.ingest(sample_pdf)
    chunks = chunker.chunk(docs)

    print(f"Pages loaded   : {len(docs)}")
    print(f"Chunks produced: {len(chunks)}")
    print(f"Stats          : {chunker.stats(chunks)}")
    print()

    for chunk in chunks[:3]:

        m = chunk.metadata

        print(
            f"  [{m['chunk_index']}/{m['chunk_total']}]  {m.get('research_paper_name', '')}  |  {len(chunk.page_content)} chars"
        )
        print(f"  Preview: {chunk.page_content[:120].strip()!r}")
        print()
