"""
Tests for RAGPipeline in autorag.rag.pipeline
"""
import pytest
from unittest.mock import Mock, patch

from autorag.rag.pipeline import RAGPipeline


class TestRAGPipelineInit:
    """Test RAGPipeline initialization."""
    
    @patch('autorag.rag.pipeline.LLMClient')
    @patch('autorag.rag.pipeline.ChromaVectorStore')
    @patch('autorag.rag.pipeline.EmbeddingService')
    def test_accepts_all_parameters(self, mock_embed, mock_store, mock_llm):
        """Should accept all configuration parameters."""
        pipeline = RAGPipeline(
            llm_provider="groq",
            llm_api_key="test-key",
            llm_model="test-model",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=256,
            chunk_overlap=25,
            collection_name="test_collection"
        )
        
        assert pipeline.chunk_size == 256
        assert pipeline.chunk_overlap == 25
    
    @patch('autorag.rag.pipeline.LLMClient')
    @patch('autorag.rag.pipeline.ChromaVectorStore')
    @patch('autorag.rag.pipeline.EmbeddingService')
    def test_default_parameters(self, mock_embed, mock_store, mock_llm):
        """Should use default parameters when not specified."""
        pipeline = RAGPipeline(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        assert pipeline.chunk_size == 500
        assert pipeline.chunk_overlap == 50


class TestRAGPipelineIndexing:
    """Test document indexing."""
    
    @patch('autorag.rag.pipeline.LLMClient')
    @patch('autorag.rag.pipeline.ChromaVectorStore')
    @patch('autorag.rag.pipeline.EmbeddingService')
    def test_index_documents_calls_upsert(self, mock_embed, mock_store, mock_llm, sample_documents):
        """index_documents should embed and upsert documents."""
        # Setup mocks
        mock_embed_instance = Mock()
        mock_embed_instance.embed_batch.return_value = [[0.1] * 384]
        mock_embed.return_value = mock_embed_instance
        
        mock_store_instance = Mock()
        mock_store.return_value = mock_store_instance
        
        pipeline = RAGPipeline(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        pipeline.index_documents(sample_documents[:1])
        
        # Verify embed_batch was called
        assert mock_embed_instance.embed_batch.called
        # Verify upsert was called
        assert mock_store_instance.upsert_documents.called
    
    @patch('autorag.rag.pipeline.LLMClient')
    @patch('autorag.rag.pipeline.ChromaVectorStore')
    @patch('autorag.rag.pipeline.EmbeddingService')
    def test_index_empty_documents(self, mock_embed, mock_store, mock_llm):
        """Should handle empty document list gracefully."""
        pipeline = RAGPipeline(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        # Should not raise
        pipeline.index_documents([])


class TestRAGPipelineQuery:
    """Test query functionality."""
    
    @patch('autorag.rag.pipeline.LLMClient')
    @patch('autorag.rag.pipeline.ChromaVectorStore')
    @patch('autorag.rag.pipeline.EmbeddingService')
    def test_query_no_documents_returns_message(self, mock_embed, mock_store, mock_llm):
        """Query with no matching documents should return appropriate message."""
        mock_embed_instance = Mock()
        mock_embed_instance.embed_text.return_value = [0.1] * 384
        mock_embed.return_value = mock_embed_instance
        
        mock_store_instance = Mock()
        mock_store_instance.search.return_value = []
        mock_store.return_value = mock_store_instance
        
        pipeline = RAGPipeline(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        result = pipeline.query("What is Python?")
        
        assert "No relevant documents" in result["answer"]
        assert result["sources"] == []
