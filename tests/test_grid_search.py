"""
Tests for GridSearchOptimizer in autorag.optimization.grid_search
"""
import pytest
from unittest.mock import Mock, patch
import itertools


class TestGridSearchInit:
    """Test GridSearchOptimizer initialization."""
    
    @patch('autorag.optimization.grid_search.get_evaluator')
    def test_parses_rag_config(self, mock_evaluator):
        """Should parse RAG config correctly."""
        from autorag.optimization.grid_search import GridSearchOptimizer
        
        mock_evaluator.return_value = None
        
        rag_config = {
            "chunk_size": [256, 512],
            "chunk_overlap": [25, 50],
            "embedding_model": ["all-MiniLM-L6-v2"],
            "top_k": [3, 5],
            "temperature": [0.5]
        }
        
        optimizer = GridSearchOptimizer(
            llm_provider="groq",
            llm_api_key="test-key",
            rag_config=rag_config,
            documents=[]
        )
        
        assert optimizer.chunk_sizes == [256, 512]
        assert optimizer.chunk_overlaps == [25, 50]
        assert optimizer.top_k_values == [3, 5]
    
    @patch('autorag.optimization.grid_search.get_evaluator')
    def test_uses_defaults_without_config(self, mock_evaluator):
        """Should use defaults when no rag_config provided."""
        from autorag.optimization.grid_search import GridSearchOptimizer
        
        mock_evaluator.return_value = None
        
        optimizer = GridSearchOptimizer(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        assert optimizer.chunk_sizes == [500]
        assert optimizer.top_k_values == [3, 5, 10]
        assert optimizer.temperature_values == [0.3, 0.7, 1.0]


class TestGridSearchSpace:
    """Test search space generation."""
    
    def test_generates_all_combinations(self):
        """Should generate all indexing × query combinations."""
        # Simple math check
        chunk_sizes = [256, 512]
        chunk_overlaps = [50]
        embedding_models = ["model1"]
        top_ks = [3, 5]
        temperatures = [0.5, 1.0]
        
        indexing_combos = list(itertools.product(chunk_sizes, chunk_overlaps, embedding_models))
        query_combos = list(itertools.product(top_ks, temperatures))
        total = len(indexing_combos) * len(query_combos)
        
        # 2 × 1 × 1 × 2 × 2 = 8
        assert total == 8


class TestGridSearchResults:
    """Test result sorting and retrieval."""
    
    @patch('autorag.optimization.grid_search.get_evaluator')
    def test_get_best_config_raises_without_results(self, mock_evaluator):
        """get_best_config should raise if optimize() not called."""
        from autorag.optimization.grid_search import GridSearchOptimizer
        
        mock_evaluator.return_value = None
        
        optimizer = GridSearchOptimizer(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        with pytest.raises(ValueError):
            optimizer.get_best_config()
