"""
Tests for text chunking utilities in autorag.utils.text_utils
"""
import pytest

from autorag.utils.text_utils import chunk_text, chunk_documents


class TestChunkText:
    """Test chunk_text function."""
    
    def test_small_text_single_chunk(self):
        """Small text should result in a single chunk."""
        text = "Short text"
        chunks = chunk_text(text, doc_id="doc1", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text"
        assert chunks[0]["id"] == "doc1_0"
    
    def test_large_text_multiple_chunks(self):
        """Large text should be split into multiple chunks."""
        text = "A" * 1000
        chunks = chunk_text(text, doc_id="doc1", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1
    
    def test_chunk_respects_size_limit(self):
        """Chunks should not exceed the size limit (with some tolerance)."""
        text = "A" * 1000
        chunks = chunk_text(text, doc_id="doc1", chunk_size=200, chunk_overlap=0)
        # Allow some tolerance for word boundaries
        for chunk in chunks:
            assert len(chunk["text"]) <= 250
    
    def test_chunk_ids_are_unique(self):
        """Each chunk should have a unique ID."""
        text = "A" * 1000
        chunks = chunk_text(text, doc_id="doc1", chunk_size=200, chunk_overlap=50)
        ids = [chunk["id"] for chunk in chunks]
        assert len(ids) == len(set(ids))
    
    def test_empty_text_returns_empty(self):
        """Empty text should return empty list or single empty chunk."""
        chunks = chunk_text("", doc_id="doc1", chunk_size=500)
        # Either empty list or single chunk with empty/whitespace text
        assert len(chunks) <= 1


class TestChunkDocuments:
    """Test chunk_documents function."""
    
    def test_chunks_multiple_documents(self, sample_documents):
        """Should chunk all documents."""
        chunks = chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= len(sample_documents)
    
    def test_preserves_metadata(self, sample_documents):
        """Chunks should preserve original document metadata."""
        chunks = chunk_documents(sample_documents, chunk_size=500)
        for chunk in chunks:
            assert "metadata" in chunk
    
    def test_empty_documents_returns_empty(self):
        """Empty document list should return empty list."""
        chunks = chunk_documents([], chunk_size=500)
        assert chunks == []
    
    def test_custom_chunk_parameters(self, sample_documents):
        """Should respect custom chunk size and overlap."""
        small_chunks = chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)
        large_chunks = chunk_documents(sample_documents, chunk_size=1000, chunk_overlap=100)
        # Smaller chunk size should produce more chunks
        assert len(small_chunks) >= len(large_chunks)
