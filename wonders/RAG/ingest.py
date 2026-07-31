"""
ingest.py
=========
DocumentIngestor: dispatches to one FormatParser (RAG/parsers/) per file
extension, then enriches whatever they return with a shared metadata
contract that RAG/chunking.py and RAG/vector_store.py depend on.

No `unstructured` package, no remote/external partitioning API anywhere in
this pipeline — see RAG/parsers/base.py's module docstring for why. This
module previously wrapped langchain_unstructured.UnstructuredLoader; that
code path is gone, but the public API (ingest / ingest_file_stream /
ingest_directory / supported_formats) is unchanged so callers (VectorStore,
tests) don't need to change.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from RAG.ocr import OcrProvider
from RAG.parsers.docx_parser import DocxParser
from RAG.parsers.markdown_parser import MarkdownParser
from RAG.parsers.pdf_parser import PdfParser
from RAG.parsers.text_parser import PlainTextParser

# ─────────────────────────────────────────────────────────────────────────────
# Extensions this pipeline can dispatch to a FormatParser. Adding a new
# format means adding one new parser class under RAG/parsers/ and one entry
# in DocumentIngestor._build_parsers() below — nothing else in the ingestion
# pipeline needs to change (Strategy pattern).
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}

# Element categories every FormatParser tags its output with. Unlike the old
# Unstructured-based filter, there's no boilerplate ("Header"/"Footer"/etc.)
# to drop here — each parser only ever emits meaningful content. "OCRText" is
# the literal text of an embedded image (see RAG/ocr.py); it's only produced
# when an ocr_provider is supplied to DocumentIngestor.
ALLOWED_ELEMENT_CATEGORIES = {"NarrativeText", "Table", "Image", "OCRText"}


class DocumentIngestor:
    """
    Dispatches to a per-format parser and enriches the result with metadata
    (source, file_name, content_type, char/word counts) that the rest of the
    pipeline relies on.

    Parameters
    ----------
    image_output_dir : str, optional
        Where extracted embedded images (from PDF/DOCX) get saved so the VLM
        `read_image` tool has a real path to look at. Defaults to the OS temp
        dir if not provided here or per-call; callers that care about
        persistence (e.g. Backend/conversation_service.py, which wants images
        to live next to the rest of a conversation's uploads) should pass one
        explicitly.
    ocr_provider : OcrProvider, optional
        Strategy for extracting literal text out of embedded images (PDF/DOCX
        only — see RAG/ocr.py). Left as None by default (no OCR run) so bare
        `DocumentIngestor()` calls (tests, RAG/main.py's CLI demo) don't pay
        for loading an OCR engine; production wires a real one in via
        Backend/deps.py -> RAG/vector_store.py's VectorStore/VectorStoreFactory.

    Usage
    -----
    ingestor = DocumentIngestor()
    docs = ingestor.ingest("path/to/document.pdf")
    docs = ingestor.ingest_directory("path/to/docs/", recursive=True)
    """

    def __init__(
        self,
        image_output_dir: Optional[str] = None,
        ocr_provider: Optional[OcrProvider] = None,
    ):
        self._image_output_dir = image_output_dir
        self._parsers = self._build_parsers(ocr_provider)

    @staticmethod
    def _build_parsers(ocr_provider: Optional[OcrProvider]) -> Dict[str, object]:
        return {
            ".txt": PlainTextParser(),
            ".md": MarkdownParser(),
            ".docx": DocxParser(ocr_provider=ocr_provider),
            ".pdf": PdfParser(ocr_provider=ocr_provider),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def ingest(self, file_path: str, image_output_dir: Optional[str] = None) -> List[Document]:
        """Ingest a single file of any supported type."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        parser = self._get_parser(path.suffix.lower())
        elements = parser.parse(
            str(path), image_output_dir=image_output_dir or self._image_output_dir or str(path.parent)
        )

        for i, doc in enumerate(elements):
            doc.metadata = self._build_metadata(path, doc, element_index=i + 1)

        return elements

    def ingest_file_stream(
        self, file_content: bytes, file_name: str, image_output_dir: Optional[str] = None
    ) -> List[Document]:
        """
        Ingest a file from memory (e.g. a FastAPI UploadFile's bytes) without
        writing the original to a permanent location first.
        """
        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {ext}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_f:
            temp_f.write(file_content)
            temp_path = temp_f.name

        try:
            parser = self._get_parser(ext)
            elements = parser.parse(
                temp_path, image_output_dir=image_output_dir or self._image_output_dir
            )

            # Build metadata, but use the ORIGINAL file name for display logic
            display_path = Path(file_name)
            for i, doc in enumerate(elements):
                doc.metadata = self._build_metadata(display_path, doc, element_index=i + 1)
                doc.metadata["source"] = "User Upload"

            return elements
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def ingest_directory(self, dir_path: str, recursive: bool = False) -> List[Document]:
        """Ingest all supported files inside a directory."""
        root = Path(dir_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        pattern = "**/*" if recursive else "*"
        all_docs: List[Document] = []

        for file in root.glob(pattern):
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs = self.ingest(str(file))
                    all_docs.extend(docs)
                    print(f"[OK] Ingested {file.name} ({len(docs)} element(s))")
                except Exception as exc:
                    print(f"[FAIL] {file.name}: {exc}")

        return all_docs

    def supported_formats(self) -> List[str]:
        """Return the list of currently supported file extensions."""
        return sorted(SUPPORTED_EXTENSIONS)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_parser(self, ext: str):
        parser = self._parsers.get(ext)
        if parser is None:
            raise ValueError(f"Unsupported extension: {ext}")
        return parser

    @staticmethod
    def _build_metadata(path: Path, doc: Document, element_index: int) -> dict:
        """Build enriched metadata dict for a single parsed element."""
        paper_name = path.stem.replace("_", " ").replace("-", " ").title()
        content = doc.page_content or ""
        category = doc.metadata.get("category", "")

        content_type = {"Image": "image", "Table": "table", "OCRText": "image_text"}.get(
            category, "text"
        )

        meta = doc.metadata.copy() if doc.metadata else {}
        meta.update(
            {
                "source": str(path),
                "file_name": path.name,
                "research_paper_name": paper_name,
                "document_type": path.suffix.lstrip(".").upper(),
                "element_index": element_index,
                "content_type": content_type,
                "content_preview": (
                    (content[:200].strip() + "…") if len(content) > 200 else content.strip()
                ),
                "char_count": len(content),
                "word_count": len(content.split()),
            }
        )
        return meta
