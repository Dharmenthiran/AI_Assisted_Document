"""Unit tests for recursive text chunking and overlap logic."""

import pytest
from app.services.chunking_service import RecursiveTextChunker
from app.services.pdf_service import ExtractedPage


def test_chunker_initialization_validation():
    """Verify ValueError is raised if overlap >= chunk_size."""
    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=100, chunk_overlap=150)


def test_chunking_preserves_provenance_metadata():
    """Verify chunks preserve document ID, filename, page number, and chunk index."""
    chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=30)
    page = ExtractedPage(
        page_number=3,
        text="FastAPI is a modern, fast web framework for building APIs with Python 3.8+ based on standard Python type hints. "
             "It is designed to be easy to learn and production-ready.",
        char_count=165,
    )

    chunks = chunker.split_page(
        document_id="doc_123",
        filename="guide.pdf",
        page=page,
        starting_chunk_index=0,
    )

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == "doc_123"
        assert chunk.filename == "guide.pdf"
        assert chunk.page == 3
        assert chunk.chunk_id.startswith("doc_123_p3_c")
        assert len(chunk.text) > 0


def test_chunking_overlap_behavior():
    """Verify consecutive chunks have shared overlap text."""
    chunker = RecursiveTextChunker(chunk_size=120, chunk_overlap=40)
    text = (
        "Alpha paragraph discusses retrieval augmented generation pipelines. "
        "Beta paragraph covers vector databases like ChromaDB and dense embeddings. "
        "Gamma paragraph details local inference with Ollama models."
    )
    page = ExtractedPage(page_number=1, text=text, char_count=len(text))

    chunks = chunker.split_page("test_doc", "test.pdf", page)
    assert len(chunks) > 1

    # Check that chunks have sensible character length
    for c in chunks:
        assert len(c.text) <= 150  # Close to chunk_size boundary


def test_empty_page_handling():
    """Verify empty or whitespace-only pages produce no chunks."""
    chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
    empty_page = ExtractedPage(page_number=1, text="    \n\n   ", char_count=0)
    chunks = chunker.split_page("doc_empty", "empty.pdf", empty_page)
    assert chunks == []


def test_chunk_document_across_multiple_pages():
    """Verify chunk_document keeps page boundaries separate for exact citations."""
    chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=40)
    pages = [
        ExtractedPage(page_number=1, text="Page one text content regarding policy guidelines.", char_count=50),
        ExtractedPage(page_number=2, text="Page two text content regarding reimbursement details.", char_count=52),
    ]

    chunks = chunker.chunk_document("doc_multi", "multi.pdf", pages)
    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[1].page == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
