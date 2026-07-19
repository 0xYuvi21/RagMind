"""
Tests for RAG/ingest.py's DocumentIngestor and the per-format parsers under
RAG/parsers/. Central contract (task spec, item D): a file with mixed content
(text + at least one table + at least one image) must produce chunks
representing all three content types — nothing gets silently dropped.
"""

from __future__ import annotations

from RAG.ingest import DocumentIngestor
from tests.fixtures.mixed_content import (
    make_mixed_docx,
    make_mixed_markdown,
    make_mixed_pdf,
    make_mixed_txt,
)


def _categories(docs):
    return {d.metadata.get("category") for d in docs}


def test_docx_mixed_content_retains_text_table_and_image(tmp_path):
    path = make_mixed_docx(tmp_path)
    docs = DocumentIngestor(image_output_dir=str(tmp_path / "images")).ingest(str(path))

    categories = _categories(docs)
    assert {"NarrativeText", "Table", "Image"} <= categories

    table_doc = next(d for d in docs if d.metadata["category"] == "Table")
    assert "Header A" in table_doc.page_content and "Value 1" in table_doc.page_content

    image_doc = next(d for d in docs if d.metadata["category"] == "Image")
    assert image_doc.metadata["content_type"] == "image"
    assert (tmp_path / "images").exists()
    assert any((tmp_path / "images").iterdir())


def test_pdf_mixed_content_retains_text_table_and_image(tmp_path):
    path = make_mixed_pdf(tmp_path)
    docs = DocumentIngestor(image_output_dir=str(tmp_path / "images")).ingest(str(path))

    categories = _categories(docs)
    assert {"NarrativeText", "Table", "Image"} <= categories

    table_doc = next(d for d in docs if d.metadata["category"] == "Table")
    assert "Header A" in table_doc.page_content and "Value 2" in table_doc.page_content


def test_markdown_mixed_content_retains_text_table_and_image(tmp_path):
    path = make_mixed_markdown(tmp_path)
    docs = DocumentIngestor().ingest(str(path))

    categories = _categories(docs)
    assert {"NarrativeText", "Table", "Image"} <= categories

    image_doc = next(d for d in docs if d.metadata["category"] == "Image")
    assert "diagram.png" in image_doc.metadata.get("image_ref", "")


def test_txt_ingestion_produces_narrative_text(tmp_path):
    path = make_mixed_txt(tmp_path)
    docs = DocumentIngestor().ingest(str(path))

    assert len(docs) == 2  # two blank-line-separated paragraphs
    assert all(d.metadata["category"] == "NarrativeText" for d in docs)
    assert all(d.metadata["content_type"] == "text" for d in docs)


def test_metadata_enrichment_on_every_element(tmp_path):
    path = make_mixed_txt(tmp_path)
    docs = DocumentIngestor().ingest(str(path))

    for doc in docs:
        assert doc.metadata["file_name"] == "mixed.txt"
        assert doc.metadata["document_type"] == "TXT"
        assert doc.metadata["char_count"] == len(doc.page_content)
        assert doc.metadata["word_count"] == len(doc.page_content.split())
        assert "element_index" in doc.metadata


def test_unsupported_extension_raises(tmp_path):
    bogus = tmp_path / "file.exe"
    bogus.write_bytes(b"not a real document")

    try:
        DocumentIngestor().ingest(str(bogus))
        assert False, "expected ValueError for unsupported extension"
    except ValueError:
        pass


def test_missing_file_raises_file_not_found():
    try:
        DocumentIngestor().ingest("does/not/exist.txt")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_ingest_file_stream_matches_ingest_for_same_content(tmp_path):
    path = make_mixed_docx(tmp_path)
    content = path.read_bytes()

    ingestor = DocumentIngestor(image_output_dir=str(tmp_path / "images2"))
    docs_from_stream = ingestor.ingest_file_stream(content, "mixed.docx")

    categories = _categories(docs_from_stream)
    assert {"NarrativeText", "Table", "Image"} <= categories
    assert all(d.metadata["source"] == "User Upload" for d in docs_from_stream)


def test_supported_formats_lists_all_four_required_formats():
    formats = DocumentIngestor().supported_formats()
    assert {".pdf", ".docx", ".txt", ".md"} <= set(formats)


def test_ingest_directory_ingests_all_supported_files(tmp_path):
    make_mixed_txt(tmp_path, name="a.txt")
    make_mixed_markdown(tmp_path, name="b.md")
    (tmp_path / "ignored.exe").write_bytes(b"binary junk")

    docs = DocumentIngestor().ingest_directory(str(tmp_path))
    file_names = {d.metadata["file_name"] for d in docs}
    assert file_names == {"a.txt", "b.md"}
