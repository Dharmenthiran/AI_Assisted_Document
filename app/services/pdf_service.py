"""PDF Validation, Storage, and Text Extraction Service.

Uses PyMuPDF (fitz) to reliably extract text page-by-page with page number preservation,
whitespace normalization, security sanitization, and scanned-PDF detection.
"""

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pymupdf  # Modern PyMuPDF API


class PDFProcessingError(Exception):
    """Base exception for PDF processing failures."""
    pass


class PDFValidationError(PDFProcessingError):
    """Raised when uploaded file fails size, format, or structure validation."""
    pass


class NoExtractableTextError(PDFProcessingError):
    """Raised when PDF contains no extractable text (e.g., scanned document or images)."""
    pass


@dataclass
class ExtractedPage:
    """Represents text extracted from a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal and shell injection.
    
    Preserves alphanumeric characters, dashes, underscores, and dots.
    """
    # Remove path components
    basename = os.path.basename(filename)
    # Strip any directory separators or dangerous patterns
    clean = re.sub(r"[^\w\s\.-]", "_", basename).strip()
    # Normalize multiple whitespace to single space
    clean = re.sub(r"\s+", "_", clean)
    if not clean or clean.startswith("."):
        clean = f"document_{clean}" if clean else "document.pdf"
    if not clean.lower().endswith(".pdf"):
        clean = f"{clean}.pdf"
    return clean


def normalize_whitespace(text: str) -> str:
    """Normalize excess whitespace, tabs, and duplicate blank lines while preserving paragraphs."""
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace vertical tabs or form feeds
    text = re.sub(r"[\x0b\x0c]", "\n", text)
    # Replace multiple spaces with single space within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Collapse more than two consecutive empty lines to two
    normalized = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                normalized.append("")
        else:
            blank_count = 0
            normalized.append(line)
    return "\n".join(normalized).strip()


class PDFService:
    """Handles PDF validation, saving, and text extraction."""

    def __init__(self, uploads_dir: str, max_upload_mb: int = 15):
        self.uploads_dir = Path(uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_upload_mb * 1024 * 1024
        self.max_upload_mb = max_upload_mb

    def validate_file(self, filename: str, content: bytes) -> str:
        """Validate file size, extension, and PDF header magic bytes.
        
        Returns the sanitized filename.
        """
        if not filename or not filename.lower().endswith(".pdf"):
            raise PDFValidationError(f"Invalid file type for '{filename}'. Only .pdf files are accepted.")

        if len(content) == 0:
            raise PDFValidationError("The uploaded file is empty (0 bytes).")

        if len(content) > self.max_bytes:
            actual_mb = round(len(content) / (1024 * 1024), 2)
            raise PDFValidationError(
                f"File size ({actual_mb} MB) exceeds maximum allowed limit of {self.max_upload_mb} MB."
            )

        # Validate PDF magic header (%PDF-)
        if not content.startswith(b"%PDF-"):
            raise PDFValidationError("Invalid file content: Missing standard PDF header (%PDF-).")

        return sanitize_filename(filename)

    def save_file(self, content: bytes, clean_filename: str) -> Tuple[str, Path]:
        """Save validated PDF bytes to uploads directory with unique document ID.
        
        Returns (document_id, file_path).
        """
        doc_id = uuid.uuid4().hex[:12]
        safe_name = f"{doc_id}_{clean_filename}"
        file_path = self.uploads_dir / safe_name
        with open(file_path, "wb") as f:
            f.write(content)
        return doc_id, file_path

    def extract_text(self, file_path: Path) -> List[ExtractedPage]:
        """Extract text page by page from the PDF file using PyMuPDF.
        
        Raises NoExtractableTextError if no extractable text is found in any page.
        """
        if not file_path.exists():
            raise PDFProcessingError(f"PDF file does not exist at {file_path}")

        extracted_pages: List[ExtractedPage] = []
        total_text_chars = 0

        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            raise PDFValidationError(f"Corrupted or malformed PDF: {str(e)}")

        try:
            if doc.page_count == 0:
                raise PDFValidationError("PDF document contains no pages.")

            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text") or ""
                cleaned_text = normalize_whitespace(raw_text)
                char_count = len(cleaned_text)
                total_text_chars += char_count

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_idx + 1,
                        text=cleaned_text,
                        char_count=char_count,
                    )
                )

        finally:
            doc.close()

        if total_text_chars == 0:
            raise NoExtractableTextError(
                "No extractable text was found in the PDF. The document may be a scanned image or "
                "raster graphic requiring Optical Character Recognition (OCR), which is not currently supported."
            )

        return extracted_pages
