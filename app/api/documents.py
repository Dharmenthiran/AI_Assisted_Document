"""Document management endpoints: upload, list, delete."""

import logging
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import (
    get_pdf_service,
    get_chunker,
    get_embedding_service,
    get_vector_store,
)
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.chunking_service import RecursiveTextChunker
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import (
    NoExtractableTextError,
    PDFProcessingError,
    PDFService,
    PDFValidationError,
)
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload and index"),
    pdf_service: PDFService = Depends(get_pdf_service),
    chunker: RecursiveTextChunker = Depends(get_chunker),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """Validate, store, extract text, chunk, embed, and index a PDF document."""
    # 1. Read file bytes
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    # 2. Validate PDF format and file size
    try:
        clean_filename = pdf_service.validate_file(file.filename or "uploaded.pdf", content)
    except PDFValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # 3. Save to disk
    try:
        doc_id, file_path = pdf_service.save_file(content, clean_filename)
    except Exception as e:
        logger.error(f"Failed to save file to disk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded document on the server.",
        )

    # 4. Extract text page by page
    try:
        extracted_pages = pdf_service.extract_text(file_path)
    except NoExtractableTextError as e:
        # Clean up file on disk if text extraction found no text
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except (PDFValidationError, PDFProcessingError) as e:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        logger.error(f"Unexpected error extracting text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while extracting text from the PDF.",
        )

    # 5. Chunk pages recursively
    try:
        chunks = chunker.chunk_document(
            document_id=doc_id,
            filename=clean_filename,
            pages=extracted_pages,
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        logger.error(f"Failed to chunk document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to segment document into text chunks.",
        )

    if not chunks:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document text could not produce valid text chunks.",
        )

    # 6. Generate embeddings for all chunks
    try:
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)
    except Exception as e:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding model error: {str(e)}",
        )

    # 7. Store chunks and register document metadata
    try:
        vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
        vector_store.register_document(
            document_id=doc_id,
            filename=clean_filename,
            file_size_bytes=len(content),
            page_count=len(extracted_pages),
            chunk_count=len(chunks),
            status="indexed",
        )
    except Exception as e:
        logger.error(f"Failed to index vector records: {e}")
        # Rollback partial insertions
        vector_store.delete_document(doc_id)
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document chunks into vector database: {str(e)}",
        )

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=clean_filename,
        status="indexed",
        page_count=len(extracted_pages),
        chunk_count=len(chunks),
        message=f"Document '{clean_filename}' successfully indexed with {len(chunks)} chunks across {len(extracted_pages)} page(s).",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
def list_documents(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentListResponse:
    """Retrieve all uploaded documents and their indexing statuses."""
    docs = vector_store.list_documents()
    return DocumentListResponse(documents=docs, total_count=len(docs))


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and purge vector records",
)
def delete_document(
    document_id: str,
    pdf_service: PDFService = Depends(get_pdf_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentDeleteResponse:
    """Purge document vector records from ChromaDB and remove physical file from disk."""
    doc_meta = vector_store.get_document(document_id)
    if not doc_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # Delete vector records
    deleted_chunks = vector_store.delete_document(document_id)

    # Delete file from disk if present
    file_pattern = f"{document_id}_*"
    for matched_file in pdf_service.uploads_dir.glob(file_pattern):
        try:
            matched_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete physical file {matched_file}: {e}")

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=doc_meta.filename,
        deleted_chunks=deleted_chunks,
        message=f"Document '{doc_meta.filename}' and its {deleted_chunks} indexed chunk(s) were successfully deleted.",
    )
