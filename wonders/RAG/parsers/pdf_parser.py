"""PdfParser — FormatParser implementation for .pdf files, via PyMuPDF (fitz).

Per-page text blocks -> NarrativeText, ruled tables (page.find_tables(), a
built-in PyMuPDF heuristic — no external model) -> Table, embedded images
(page.get_images() + doc.extract_image()) -> Image (raw bytes saved to
image_output_dir for the VLM tool to use).

Text-layer only: there is no OCR step here (that would need Tesseract, an
external system binary this project deliberately avoids — see
RAG/parsers/base.py's module docstring). A scanned/image-only PDF with no
extractable text layer will yield no NarrativeText; this is a documented
limitation, not a silent failure (see CLAUDE.md).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import fitz
from langchain_core.documents import Document


class PdfParser:
    def parse(self, file_path: str, image_output_dir: Optional[str] = None) -> List[Document]:
        output_dir = Path(image_output_dir or tempfile.gettempdir())
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem

        documents: List[Document] = []
        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                documents.extend(self._extract_tables(page, page_number))
                documents.extend(self._extract_text_blocks(page, page_number))
                documents.extend(self._extract_images(pdf, page, page_number, stem, output_dir))

        return documents

    @staticmethod
    def _extract_text_blocks(page, page_number: int) -> List[Document]:
        documents = []
        for block in page.get_text("blocks"):
            text = block[4].strip() if len(block) > 4 else ""
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"category": "NarrativeText", "page_number": page_number},
                    )
                )
        return documents

    @staticmethod
    def _extract_tables(page, page_number: int) -> List[Document]:
        documents = []
        try:
            found_tables = page.find_tables()
        except Exception:  # pragma: no cover - defensive against pymupdf quirks
            return documents

        for table in found_tables.tables:
            rows = table.extract()
            table_text = "\n".join("\t".join(cell or "" for cell in row) for row in rows)
            if table_text.strip():
                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={"category": "Table", "page_number": page_number},
                    )
                )
        return documents

    @staticmethod
    def _extract_images(pdf, page, page_number: int, stem: str, output_dir: Path) -> List[Document]:
        documents = []
        for index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                info = pdf.extract_image(xref)
            except Exception:  # pragma: no cover - defensive against malformed images
                continue

            image_path = output_dir / f"{stem}_p{page_number}_image_{index}.{info['ext']}"
            image_path.write_bytes(info["image"])
            documents.append(
                Document(
                    page_content=(
                        f"[Image on page {page_number} — use the read_image tool on "
                        f"{image_path} for a visual description.]"
                    ),
                    metadata={
                        "category": "Image",
                        "page_number": page_number,
                        "image_path": str(image_path),
                    },
                )
            )
        return documents
