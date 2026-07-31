"""
OcrProvider — extracts the exact text content of a raster image, independent
of VLM/imagellm.py's `read_image` tool (which describes an image's visual
content on-demand, not its literal text). Used by RAG/parsers/pdf_parser.py
and docx_parser.py to turn an embedded image into a real, retrievable
NarrativeText-equivalent chunk instead of only a "[go look at this image]"
placeholder.

RapidOcrProvider wraps rapidocr-onnxruntime: pure-Python + onnxruntime, no
system binaries (Tesseract/poppler) required — consistent with this
project's no-native-deps constraint (see CLAUDE.md's "Dropped `unstructured`"
section). Which OCR backend is active is controlled by config.yaml
(`ocr_provider`, `ocr_enabled`), not by editing this file.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OcrProvider(Protocol):
    """Strategy for pulling literal text out of an image file."""

    def extract_text(self, image_path: str) -> str: ...


class RapidOcrProvider:
    """OcrProvider implementation backed by RapidOCR. The onnxruntime engine
    (det/cls/rec ONNX models) is loaded lazily on first use and cached, since
    constructing it is the expensive part — not every RapidOcrProvider()
    instantiation."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def extract_text(self, image_path: str) -> str:
        result, _elapse = self._get_engine()(image_path)
        if not result:
            return ""
        return "\n".join(line[1] for line in result)
