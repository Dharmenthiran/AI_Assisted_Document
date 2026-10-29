"""Unit tests for PDF validation and text extraction."""

import pytest
from app.services.pdf_service import (
    PDFService,
    PDFValidationError,
    NoExtractableTextError,
    sanitize_filename,
    normalize_whitespace,
)


def test_sanitize_filename():
    """Verify dangerous path traversal and special characters are stripped."""
    assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("..\\..\\windows\\system32.pdf") == "system32.pdf"
    assert sanitize_filename("my report (final) 2026.pdf") == "my_report__final__2026.pdf"
    assert sanitize_filename("noextension") == "noextension.pdf"


def test_normalize_whitespace():
    """Verify line breaks and excessive whitespace normalization."""
    raw = "Line 1   with   spaces\r\n\r\n\r\n\r\nLine 2\n\n\n"
    normalized = normalize_whitespace(raw)
    assert "   " not in normalized
    assert "\r" not in normalized
    assert "Line 1 with spaces" in normalized
    assert "Line 2" in normalized


def test_pdf_validation_extension_and_size(test_temp_dir):
    """Verify non-pdf extensions and size limit enforcement."""
    service = PDFService(uploads_dir=str(test_temp_dir), max_upload_mb=1)

    # Invalid extension
    with pytest.raises(PDFValidationError, match="Only .pdf files are accepted"):
        service.validate_file("document.docx", b"some bytes")

    # Empty bytes
    with pytest.raises(PDFValidationError, match="uploaded file is empty"):
        service.validate_file("empty.pdf", b"")

    # Oversized file
    big_content = b"%PDF-" + b"0" * (2 * 1024 * 1024)  # 2MB > 1MB limit
    with pytest.raises(PDFValidationError, match="exceeds maximum allowed limit"):
        service.validate_file("big.pdf", big_content)


def test_pdf_validation_magic_bytes(test_temp_dir):
    """Verify that files missing standard %PDF- header are rejected."""
    service = PDFService(uploads_dir=str(test_temp_dir), max_upload_mb=10)
    fake_pdf = b"NOT_A_REAL_PDF_HEADER_CONTENT"
    with pytest.raises(PDFValidationError, match="Missing standard PDF header"):
        service.validate_file("fake.pdf", fake_pdf)


def test_pdf_save_and_extract_valid(test_temp_dir, sample_pdf_bytes):
    """Verify saving and extracting text from a valid generated PDF."""
    service = PDFService(uploads_dir=str(test_temp_dir), max_upload_mb=10)

    clean_name = service.validate_file("sample_policy.pdf", sample_pdf_bytes)
    assert clean_name == "sample_policy.pdf"

    doc_id, file_path = service.save_file(sample_pdf_bytes, clean_name)
    assert file_path.exists()
    assert doc_id in file_path.name

    pages = service.extract_text(file_path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "15 days of annual leave" in pages[0].text
    assert pages[1].page_number == 2
    assert "medical, dental, and vision" in pages[1].text


def test_pdf_extract_empty_or_scanned_raises(test_temp_dir, empty_pdf_bytes):
    """Verify NoExtractableTextError is raised when PDF has no text characters."""
    service = PDFService(uploads_dir=str(test_temp_dir), max_upload_mb=10)
    clean_name = service.validate_file("scanned.pdf", empty_pdf_bytes)
    doc_id, file_path = service.save_file(empty_pdf_bytes, clean_name)

    with pytest.raises(NoExtractableTextError, match="No extractable text was found"):
        service.extract_text(file_path)
