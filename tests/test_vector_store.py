"""
Tests for ChromaVectorStore - basic unit tests without actual ChromaDB.
These tests verify the interface without heavy ChromaDB operations.
"""
import pytest
from unittest.mock import Mock, patch


class TestChromaVectorStoreInterface:
    """Test ChromaVectorStore interface using mocks."""
    
    @patch('autorag.rag.chroma_store.chromadb.PersistentClient')
    def test_init_creates_collection(self, mock_client):
        """Should create collection on init."""
        from autorag.rag.chroma_store import ChromaVectorStore
        
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(collection_name="test")
        
        assert store.collection == mock_collection
        mock_client.return_value.get_or_create_collection.assert_called_once()
    
    @patch('autorag.rag.chroma_store.chromadb.PersistentClient')
    def test_count_returns_collection_count(self, mock_client):
        """count() should return collection count."""
        from autorag.rag.chroma_store import ChromaVectorStore
        
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(collection_name="test")
        
        assert store.count() == 42
    
    @patch('autorag.rag.chroma_store.chromadb.PersistentClient')
    def test_get_stats_returns_dict(self, mock_client):
        """get_stats() should return stats dict."""
        from autorag.rag.chroma_store import ChromaVectorStore
        
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(collection_name="mytest")
        stats = store.get_stats()
        
        assert stats["total_vector_count"] == 10
        assert stats["collection_name"] == "mytest"
    
    @patch('autorag.rag.chroma_store.chromadb.PersistentClient')
    def test_upsert_documents_calls_collection(self, mock_client):
        """upsert_documents should call collection.upsert."""
        from autorag.rag.chroma_store import ChromaVectorStore
        
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(collection_name="test")
        
        docs = [{"id": "1", "text": "hello", "metadata": {"source": "test"}}]
        embeddings = [[0.1] * 384]
        
        store.upsert_documents(docs, embeddings)
        
        assert mock_collection.upsert.called
    
    @patch('autorag.rag.chroma_store.chromadb.PersistentClient')
    def test_search_returns_formatted_results(self, mock_client):
        """search should return formatted results."""
        from autorag.rag.chroma_store import ChromaVectorStore
        
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "distances": [[0.1]],
            "documents": [["hello world"]],
            "metadatas": [[{"source": "test"}]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(collection_name="test")
        results = store.search([0.1] * 384, top_k=1)
        
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "hello world"
        assert results[0]["score"] == 0.9  # 1 - 0.1
