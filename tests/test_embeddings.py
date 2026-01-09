"""
Tests for EmbeddingService in autorag.rag.embeddings
Uses module-scoped fixture to load the model only ONCE.
"""
import pytest


# Module-scoped fixture - loads model only once for all tests
@pytest.fixture(scope="module")
def embedding_service():
    """Create embedding service once for all tests in this module."""
    from autorag.rag.embeddings import EmbeddingService
    return EmbeddingService()


class TestEmbeddingServiceInit:
    """Test EmbeddingService initialization."""
    
    def test_default_model(self, embedding_service):
        """Should use all-MiniLM-L6-v2 by default."""
        assert embedding_service.get_model_name() == "all-MiniLM-L6-v2"
    
    def test_default_dimension_384(self, embedding_service):
        """Default model should have 384 dimensions."""
        assert embedding_service.get_dimension() == 384


class TestEmbeddingServiceEmbed:
    """Test embedding generation."""
    
    def test_embed_text_returns_list(self, embedding_service):
        """embed_text should return a list of floats."""
        vec = embedding_service.embed_text("Hello world")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
    
    def test_embed_text_correct_dimension(self, embedding_service):
        """embed_text should return vector of correct dimension."""
        vec = embedding_service.embed_text("Hello world")
        assert len(vec) == embedding_service.get_dimension()
    
    def test_embed_batch_returns_list_of_lists(self, embedding_service):
        """embed_batch should return list of embedding vectors."""
        texts = ["Hello", "World"]
        vecs = embedding_service.embed_batch(texts)
        assert isinstance(vecs, list)
        assert len(vecs) == 2
        assert all(len(v) == embedding_service.get_dimension() for v in vecs)
    
    def test_embed_empty_text(self, embedding_service):
        """Should handle empty text without error."""
        vec = embedding_service.embed_text("")
        assert len(vec) == embedding_service.get_dimension()
    
    def test_embed_batch_empty_list(self, embedding_service):
        """Should handle empty batch without error."""
        vecs = embedding_service.embed_batch([])
        assert vecs == []
