"""Pytest fixtures and test helpers for Document AI Assistant."""

import io
from pathlib import Path
import tempfile
import pytest
import pymupdf  # Modern PyMuPDF API
from fastapi.testclient import TestClient

from app.main import app
from app.config import Settings
from app.dependencies import get_settings


@pytest.fixture
def test_temp_dir():
    """Create a temporary directory for test uploads and vector DB."""
    # ignore_cleanup_errors=True prevents Windows PermissionError on ChromaDB memory-mapped files during teardown
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_settings(test_temp_dir):
    """Provide isolated test settings pointing to temporary paths."""
    uploads_dir = test_temp_dir / "uploads"
    chroma_dir = test_temp_dir / "chroma"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        APP_NAME="Document AI Assistant (Test)",
        UPLOADS_DIR=str(uploads_dir),
        CHROMA_PERSIST_DIR=str(chroma_dir),
        COLLECTION_NAME="test_collection",
        CHUNK_SIZE=300,
        CHUNK_OVERLAP=50,
        TOP_K=2,
    )


@pytest.fixture
def sample_pdf_bytes():
    """Generate a clean 2-page PDF in-memory using PyMuPDF."""
    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Company Leave Policy.\n\n"
        "Employees are granted 15 days of annual leave and 12 sick leaves per calendar year. "
        "Leave requests must be submitted at least two weeks prior to the intended start date.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Health and Wellness Benefits.\n\n"
        "All full-time employees are eligible for comprehensive medical, dental, and vision insurance. "
        "The company covers 80% of insurance premiums for dependents.",
    )

    pdf_stream = io.BytesIO()
    doc.save(pdf_stream)
    doc.close()
    return pdf_stream.getvalue()


@pytest.fixture
def empty_pdf_bytes():
    """Generate an empty PDF page with zero text content (simulates scanned/empty page)."""
    doc = pymupdf.open()
    doc.new_page()  # Page exists, but has no text inserted
    pdf_stream = io.BytesIO()
    doc.save(pdf_stream)
    doc.close()
    return pdf_stream.getvalue()


@pytest.fixture
def client(test_settings):
    """FastAPI TestClient with overridden settings."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
