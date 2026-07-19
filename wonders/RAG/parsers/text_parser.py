"""PlainTextParser — FormatParser implementation for .txt files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


class PlainTextParser:
    """One NarrativeText element per blank-line-separated paragraph, so
    downstream chunking (RAG/chunking.py) still gets sensibly sized inputs
    even for one large file."""

    def parse(self, file_path: str, image_output_dir: Optional[str] = None) -> List[Document]:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [
            Document(page_content=paragraph, metadata={"category": "NarrativeText"})
            for paragraph in paragraphs
        ]
