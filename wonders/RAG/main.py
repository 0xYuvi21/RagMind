"""
main.py
=======
Entry-point for the RAG ingestion pipeline.

Wires together:
    1. ingest.py       – DocumentIngestor   : loads PDF / TXT files → LangChain Documents
    2. chunking.py     – DocumentChunker    : splits Documents into overlapping chunks
    3. vector_store.py – VectorStore        : embeds chunks and stores them in ChromaDB

Usage (run as a module from the project root, since RAG is now a package —
see CLAUDE.md)
-----
# Ingest a single file
uv run python -m RAG.main --file path/to/paper.pdf

# Ingest a whole folder (recursive)
uv run python -m RAG.main --dir path/to/docs/ --recursive

# Query after ingestion
uv run python -m RAG.main --file path/to/paper.pdf --query "What is RAG?"

# Just query an already-populated store
uv run python -m RAG.main --query "What is RAG?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Pipeline imports ────────────────────────────────────────────────────────
from RAG.ingest import DocumentIngestor
from RAG.chunking import DocumentChunker
from RAG.vector_store import VectorStore, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION

# ── Pipeline defaults ───────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5


# ════════════════════════════════════════════════════════════════════════════
# Core pipeline function
# ════════════════════════════════════════════════════════════════════════════


def run_pipeline(
    file_path: str | None = None,
    dir_path: str | None = None,
    recursive: bool = False,
    query: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source_tag: str | None = None,
) -> None:
    """
    Full ingestion + (optional) retrieval pipeline.

    Parameters
    ----------
    file_path    : path to a single document to ingest.
    dir_path     : path to a directory of documents to ingest.
    recursive    : whether to recurse into subdirectories.
    query        : if provided, run a similarity search after ingestion.
    top_k        : number of results to return for a query.
    persist_dir  : where ChromaDB stores its data on disk.
    collection   : ChromaDB collection name.
    chunk_size   : characters per chunk.
    chunk_overlap: overlapping characters between adjacent chunks.
    source_tag   : optional tag added to every chunk's metadata.
    """

    # ── Step 1 : Ingestor ────────────────────────────────────────────────
    ingestor = DocumentIngestor()
    print(f"\n{'='*60}")
    print(f"  RAG Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"  Supported formats : {ingestor.supported_formats()}")
    print(f"  Collection        : {collection}")
    print(f"  Persist dir       : {persist_dir}")
    print(f"  Chunk size        : {chunk_size}  |  overlap: {chunk_overlap}")
    print(f"{'='*60}\n")

    # ── Step 2 : Chunker ─────────────────────────────────────────────────
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ── Step 3 : VectorStore ─────────────────────────────────────────────
    store = VectorStore(
        persist_dir=persist_dir,
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ── Ingestion ────────────────────────────────────────────────────────
    total_added = 0

    if file_path:
        print(f"\n[Pipeline] Ingesting file: {file_path}")
        docs = ingestor.ingest(file_path)
        chunks = chunker.chunk(docs)

        print(f"[Pipeline] Pages loaded  : {len(docs)}")
        print(f"[Pipeline] Chunks created: {len(chunks)}")
        print(f"[Pipeline] Chunk stats   : {chunker.stats(chunks)}")

        if source_tag:
            for c in chunks:
                c.metadata["source_tag"] = source_tag

        total_added += store._upsert(chunks)

    if dir_path:
        print(f"\n[Pipeline] Ingesting directory: {dir_path}  (recursive={recursive})")
        docs = ingestor.ingest_directory(dir_path, recursive=recursive)
        chunks = chunker.chunk(docs)

        print(f"[Pipeline] Total pages   : {len(docs)}")
        print(f"[Pipeline] Total chunks  : {len(chunks)}")
        print(f"[Pipeline] Chunk stats   : {chunker.stats(chunks)}")

        if source_tag:
            for c in chunks:
                c.metadata["source_tag"] = source_tag

        total_added += store._upsert(chunks)

    if total_added:
        print(f"\n[Pipeline] ✓ Stored {total_added} chunks.")
        print(f"[Pipeline] Total vectors in '{collection}': {store.count()}")

    # ── Query (optional) ─────────────────────────────────────────────────
    if query:
        print(f"\n{'─'*60}")
        print(f"  Query : {query!r}")
        print(f"{'─'*60}")

        results = store.similarity_search_with_score(query, k=top_k)

        if not results:
            print("  No results found.")
        else:
            for rank, (doc, score) in enumerate(results, start=1):
                m = doc.metadata
                print(
                    f"\n  [{rank}] score={score:.4f}"
                    f"  |  {m.get('research_paper_name', 'N/A')}"
                    f"  |  chunk {m.get('chunk_index', '?')}/{m.get('chunk_total', '?')}"
                )
                print(f"       {doc.page_content[:200].strip()!r}")
        print()


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RAG ingestion pipeline: ingest → chunk → embed → store → query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Ingestion sources
    src = parser.add_argument_group(
        "Ingestion source (at least one required for ingestion)"
    )
    src.add_argument(
        "--file", metavar="PATH", help="Path to a single PDF or TXT file to ingest."
    )
    src.add_argument(
        "--dir", metavar="PATH", help="Path to a directory of documents to ingest."
    )
    src.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into sub-folders when --dir is used.",
    )

    # Query
    parser.add_argument(
        "--query", "-q", metavar="TEXT", help="Run a similarity search after ingestion."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K}).",
    )

    # Store config
    store = parser.add_argument_group("Vector store")
    store.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="ChromaDB persistence directory.",
    )
    store.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name."
    )

    # Chunking config
    chunk = parser.add_argument_group("Chunking")
    chunk.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Max characters per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    chunk.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )

    # Metadata
    parser.add_argument(
        "--tag",
        metavar="LABEL",
        help="Optional source_tag added to every chunk's metadata.",
    )

    return parser


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # Require at least one of --file / --dir / --query
    if not args.file and not args.dir and not args.query:
        parser.print_help()
        sys.exit(1)

    run_pipeline(
        file_path=args.file,
        dir_path=args.dir,
        recursive=args.recursive,
        query=args.query,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        source_tag=args.tag,
    )
