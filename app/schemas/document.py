"""Pydantic models for Document management and indexing."""

from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata representing an uploaded and indexed document."""

    document_id: str = Field(..., description="Unique generated document identifier")
    filename: str = Field(..., description="Original sanitized filename")
    file_size_bytes: int = Field(..., description="Size of the uploaded PDF in bytes")
    page_count: int = Field(..., description="Total pages extracted from the PDF")
    chunk_count: int = Field(..., description="Number of text chunks generated and indexed")
    status: str = Field(default="indexed", description="Processing status: indexed, failed, processing")
    created_at: str = Field(..., description="ISO 8601 timestamp when uploaded")
    error_message: Optional[str] = Field(default=None, description="Details if processing failed")


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful PDF upload and indexing."""

    document_id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    """Response returned when querying uploaded documents."""

    documents: List[DocumentMetadata]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    """Response returned when deleting a document and its vector records."""

    document_id: str
    filename: str
    deleted_chunks: int
    message: str
