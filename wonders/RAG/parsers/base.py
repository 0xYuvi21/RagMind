"""
FormatParser — one implementation per supported file extension.

No `unstructured` package and no external/remote partitioning API anywhere
in this project: each format is parsed with a small, focused library already
in the dependency tree (PyMuPDF for PDF, python-docx for DOCX, plain file I/O
for TXT/MD). This avoids the native-dependency chain `unstructured` pulls in
(poppler, tesseract, torch, onnxruntime, libmagic) — see CLAUDE.md for the
concrete failures (a libmagic segfault on plain .txt files, and hi_res PDF
partitioning needing system binaries not present in dev environments) that
motivated dropping it in favor of these dedicated parsers.

Every parser returns raw LangChain Documents tagged with a `category` in
metadata — "NarrativeText", "Table", or "Image" — the same three-way
vocabulary the rest of the pipeline (RAG/ingest.py's DocumentIngestor,
RAG/chunking.py, RAG/vector_store.py) already expects. RAG/ingest.py is the
only caller of these parsers; it dispatches by extension (FORMAT_PARSERS)
and enriches whatever they return with the shared metadata contract
(source, file_name, chunk-ready fields).
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from langchain_core.documents import Document


class FormatParser(Protocol):
    def parse(self, file_path: str, image_output_dir: Optional[str] = None) -> List[Document]: ...
