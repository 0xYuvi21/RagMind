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
metadata — "NarrativeText", "Table", "Image", or "OCRText" — the vocabulary
the rest of the pipeline (RAG/ingest.py's DocumentIngestor, RAG/chunking.py,
RAG/vector_store.py) already expects. RAG/ingest.py is the only caller of
these parsers; it dispatches by extension (FORMAT_PARSERS) and enriches
whatever they return with the shared metadata contract (source, file_name,
chunk-ready fields).

"OCRText" is emitted alongside "Image" when a parser is given an OcrProvider
(RAG/ocr.py) — it holds the image's literal extracted text (via RapidOCR),
distinct from "Image"'s placeholder pointing the VLM `read_image` tool at the
file for a visual *description*. Only PdfParser/DocxParser take an
ocr_provider constructor arg today (the two formats that embed raster
images); PlainTextParser/MarkdownParser have nothing to OCR.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from langchain_core.documents import Document


class FormatParser(Protocol):
    def parse(self, file_path: str, image_output_dir: Optional[str] = None) -> List[Document]: ...
