"""MarkdownParser — FormatParser implementation for .md files.

Lightweight structural detection (no external Markdown library needed): a
blank-line-separated block is tagged Table if it looks like a GFM table
(a `|` header row followed by a `---` separator row), Image if it's a single
image reference (`![alt](path)`), NarrativeText otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

_IMAGE_PATTERN = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


class MarkdownParser:
    def parse(self, file_path: str, image_output_dir: Optional[str] = None) -> List[Document]:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

        documents: List[Document] = []
        for block in blocks:
            image_match = _IMAGE_PATTERN.match(block)
            if image_match:
                alt_text, image_ref = image_match.groups()
                documents.append(
                    Document(
                        page_content=f"[Image: {alt_text or image_ref}]",
                        metadata={"category": "Image", "image_ref": image_ref},
                    )
                )
                continue

            if _looks_like_markdown_table(block.splitlines()):
                documents.append(Document(page_content=block, metadata={"category": "Table"}))
                continue

            documents.append(Document(page_content=block, metadata={"category": "NarrativeText"}))

        return documents


def _looks_like_markdown_table(lines: List[str]) -> bool:
    if len(lines) < 2:
        return False
    header, separator = lines[0], lines[1]
    return "|" in header and "-" in separator and re.fullmatch(r"[\s|:-]+", separator) is not None
