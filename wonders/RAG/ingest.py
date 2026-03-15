import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# File extensions that UnstructuredLoader can handle natively.
# This list is informational — UnstructuredLoader auto-detects formats on its
# own; we use it only to filter which files we try to ingest from a directory.
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    # Spreadsheets
    ".xls",
    ".xlsx",
    ".ods",
    ".csv",
    ".tsv",
    # Presentations
    ".ppt",
    ".pptx",
    # Plain text / markup
    ".txt",
    ".md",
    ".rst",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".eml",
    ".msg",
    # Images (OCR — works on scanned / handwritten content too)
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".webp",
    # E-books / other
    ".epub",
}

# ─────────────────────────────────────────────────────────────────────────────
# Element categories to keep. Unstructured categorizes every extracted chunk.
# We drop "Header", "Footer", "Title", "PageBreak", "UncategorizedText", etc.,
# to ensure only meaningful, dense content goes into the vector store.
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_ELEMENT_CATEGORIES = {
    "NarrativeText",
    "Text",
    "Table",
    "Formula",
    "FigureCaption",
    "Image",  # (If extracted explicitly)
}


class DocumentIngestor:
    """
    A modular document ingestor built on top of LangChain's UnstructuredLoader.

    Supports virtually every common document format including PDFs, Word docs,
    spreadsheets, presentations, plain-text files, HTML, e-mails, and images
    (scanned or handwritten — Unstructured uses OCR automatically).

    Parameters
    ----------
    api_key : str, optional
        Unstructured API key for remote partitioning via the serverless API.
        When omitted the loader runs fully locally (requires the open-source
        ``unstructured`` package and its system dependencies).
    api_url : str, optional
        Override the default Unstructured API endpoint.
    strategy : str
        Partitioning strategy forwarded to Unstructured.
        - ``"auto"``   – Unstructured chooses the best strategy per file type
                         (fast for text PDFs, hi_res for images). **Default.**
        - ``"hi_res"`` – Highest quality; uses layout-detection models + OCR.
                         Required for scanned PDFs and handwritten images.
        - ``"fast"``   – Text-only PDFs, skips OCR entirely.
        Defaults to ``"auto"`` — best for mixed workloads.
    extra_kwargs : dict, optional
        Any additional kwargs passed straight through to UnstructuredLoader
        (e.g. ``chunking_strategy``, ``max_characters``, etc.).

    Usage
    -----
    # --- Local (open-source) ---
    ingestor = DocumentIngestor()

    # --- Remote API ---
    ingestor = DocumentIngestor(api_key="your-unstructured-api-key")

    # Single file  (PDF, image, Word doc, …)
    docs = ingestor.ingest("path/to/document.pdf")
    docs = ingestor.ingest("path/to/handwritten_note.jpg")

    # Whole directory (recursive)
    docs = ingestor.ingest_directory("path/to/docs/", recursive=True)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        strategy: str = "auto",
        extra_kwargs: Optional[dict] = None,
    ):
        self._api_key = api_key or os.getenv("UNSTRUCTURED_API_KEY")
        self._api_url = api_url
        self._strategy = strategy
        self._extra_kwargs = extra_kwargs or {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_file_stream(self, file_content: bytes, file_name: str) -> List[Document]:
        """
        Dynamically ingest a file from memory (e.g. uploaded via Streamlit/Gradio)
        without having to save it to disk first.

        Parameters
        ----------
        file_content : bytes
            Raw byte content of the uploaded file.
        file_name : str
            Filename including extension (e.g., "report.pdf"). Ex: user_upload.name

        Returns
        -------
        List[Document]
            Extracted, meaningful LangChain chunks.
        """
        # We write it to a temporary location so UnstructuredLoader can parse it.
        # (Though if using raw Unstructured API, we could pass bytes directly.)
        import tempfile

        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {ext}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_f:
            temp_f.write(file_content)
            temp_path = temp_f.name

        try:
            # Re-use our existing ingestion logic but override the path handling
            loader = self._build_loader(temp_path)
            all_elements = loader.load()

            meaningful_documents = []
            for doc in all_elements:
                category = doc.metadata.get("category", "")
                if category in ALLOWED_ELEMENT_CATEGORIES:
                    meaningful_documents.append(doc)

            # Build metadata, but use the ORIGINAL file name for the display logic
            dummy_path = Path(file_name)
            for i, doc in enumerate(meaningful_documents):
                doc.metadata = self._build_metadata(
                    dummy_path, doc, element_index=i + 1
                )
                doc.metadata["source"] = "User Upload"

            return meaningful_documents

        finally:
            # Clean up the temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def ingest(self, file_path: str) -> List[Document]:
        """
        Ingest a single file of any supported type.

        Returns
        -------
        List[Document]
            LangChain Document objects with enriched metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        loader = self._build_loader(str(path))
        all_elements = loader.load()

        # 1. Filter out low-value elements
        meaningful_documents = []
        for doc in all_elements:
            category = doc.metadata.get("category", "")
            if category in ALLOWED_ELEMENT_CATEGORIES:
                meaningful_documents.append(doc)

        # 2. Attach enriched metadata
        for i, doc in enumerate(meaningful_documents):
            doc.metadata = self._build_metadata(path, doc, element_index=i + 1)

        return meaningful_documents

    def ingest_directory(
        self,
        dir_path: str,
        recursive: bool = False,
    ) -> List[Document]:
        """
        Ingest all supported files inside a directory.

        Parameters
        ----------
        dir_path  : str  – path to the directory
        recursive : bool – if True, walks all subdirectories

        Returns
        -------
        List[Document]
        """
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
                    print(f"[✓] Ingested  {file.name}  ({len(docs)} element(s))")
                except Exception as exc:
                    print(f"[✗] Failed    {file.name}: {exc}")

        return all_docs

    def supported_formats(self) -> List[str]:
        """Return the list of currently supported file extensions."""
        return sorted(SUPPORTED_EXTENSIONS)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_loader(self, file_path: str) -> UnstructuredLoader:
        """Construct an UnstructuredLoader for the given file."""
        kwargs: dict = {
            "strategy": self._strategy,
            **self._extra_kwargs,
        }

        if self._api_key:
            # Remote partitioning via the Unstructured serverless API
            kwargs["api_key"] = self._api_key
            kwargs["partition_via_api"] = True
            if self._api_url:
                kwargs["url"] = self._api_url

        return UnstructuredLoader(file_path, **kwargs)

    @staticmethod
    def _build_metadata(path: Path, doc: Document, element_index: int) -> dict:
        """Build enriched metadata dict for a single document element."""
        paper_name = path.stem.replace("_", " ").replace("-", " ").title()
        content = doc.page_content or ""

        meta = doc.metadata.copy() if doc.metadata else {}
        meta.update(
            {
                "source": str(path),
                "file_name": path.name,
                "research_paper_name": paper_name,
                "document_type": path.suffix.lstrip(".").upper(),
                "element_index": element_index,
                "content_preview": (
                    (content[:200].strip() + "…")
                    if len(content) > 200
                    else content.strip()
                ),
                "char_count": len(content),
                "word_count": len(content.split()),
            }
        )
        return meta


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo / manual test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # The API key is loaded from the .env file automatically
    ingestor = DocumentIngestor(strategy="auto")
    print("Supported formats:", ingestor.supported_formats())

    # ── Test with a PDF ──────────────────────────────────────────────────────
    sample_pdf = "C:/Users/muthi/Desktop/wonders/RAG/docs/CHATBOT_Architecture_Design_and_Developm.pdf"
    try:
        docs = ingestor.ingest(sample_pdf)
        print(f"\n[PDF] {len(docs)} element(s) extracted.")
        print("Total documents retrieved from the files: ", len(docs))
        print("Example doc: /n", docs[0])

    except Exception as e:
        print(f"[PDF] Error: {e}")

    # ── Test with an image (scanned or handwritten) ──────────────────────────
    # sample_image = "C:/Users/muthi/Desktop/wonders/RAG Docs/Pill Images/azithromycin.jpg"
    # try:
    #     docs = ingestor.ingest(sample_image)
    #     print(f"\n[Image] {len(docs)} element(s) extracted.")
    #     print(docs[0].page_content)
    # except Exception as e:
    #     print(f"[Image] Error: {e}")
