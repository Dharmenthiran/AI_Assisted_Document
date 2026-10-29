"""Recursive Text Chunking Service.

Splits document text into overlapping chunks using a hierarchical set of delimiters
(paragraphs, newlines, sentences, words), preserving page numbers and source metadata.
Chunks are kept page-specific so that citations always map precisely to original PDF pages.
"""

from dataclasses import dataclass
from typing import List
from app.services.pdf_service import ExtractedPage


@dataclass
class DocumentChunk:
    """Represents a text chunk with complete provenance metadata."""
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    chunk_index: int
    char_length: int


class RecursiveTextChunker:
    """Splits text recursively using delimiters to keep chunks within target size."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: List[str] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator preserving content."""
        if not separator:
            return list(text)
        return text.split(separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge smaller splits into chunks respecting chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s)
            sep_len = len(separator) if current_doc else 0

            if total_len + s_len + sep_len <= self.chunk_size:
                current_doc.append(s)
                total_len += s_len + sep_len
            else:
                if current_doc:
                    chunk = separator.join(current_doc).strip()
                    if chunk:
                        docs.append(chunk)

                    # Backtrack to satisfy overlap
                    while current_doc and total_len > self.chunk_overlap:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + (len(separator) if current_doc else 0)

                current_doc.append(s)
                total_len = sum(len(x) for x in current_doc) + (len(separator) * max(0, len(current_doc) - 1))

        if current_doc:
            chunk = separator.join(current_doc).strip()
            if chunk:
                docs.append(chunk)

        return docs

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively attempt splitting on progressively finer separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = self._split_text_with_separator(text, separator)
        good_splits: List[str] = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # Hard truncate if no finer separator
                    for start in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        chunk = s[start:start + self.chunk_size].strip()
                        if chunk:
                            final_chunks.append(chunk)
                else:
                    deeper_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(deeper_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return [c for c in final_chunks if c.strip()]

    def split_page(
        self,
        document_id: str,
        filename: str,
        page: ExtractedPage,
        starting_chunk_index: int = 0,
    ) -> List[DocumentChunk]:
        """Split a single ExtractedPage into chunks with provenance metadata."""
        if not page.text or not page.text.strip():
            return []

        raw_chunks = self._recursive_split(page.text, self.separators)
        chunks: List[DocumentChunk] = []

        for i, text_chunk in enumerate(raw_chunks):
            chunk_idx = starting_chunk_index + i
            chunk_id = f"{document_id}_p{page.page_number}_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page=page.page_number,
                    text=text_chunk,
                    chunk_index=chunk_idx,
                    char_length=len(text_chunk),
                )
            )

        return chunks

    def chunk_document(
        self,
        document_id: str,
        filename: str,
        pages: List[ExtractedPage],
    ) -> List[DocumentChunk]:
        """Chunk all pages of a document while preserving 1-indexed page boundaries."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            page_chunks = self.split_page(
                document_id=document_id,
                filename=filename,
                page=page,
                starting_chunk_index=global_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks
