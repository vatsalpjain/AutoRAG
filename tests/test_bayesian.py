"""
Tests for BayesianOptimizer in autorag.optimization.bayesian
"""
import pytest
from unittest.mock import Mock, patch


class TestBayesianInit:
    """Test BayesianOptimizer initialization."""
    
    @patch('autorag.optimization.bayesian.get_evaluator')
    def test_parses_rag_config(self, mock_evaluator):
        """Should parse RAG config correctly."""
        from autorag.optimization.bayesian import BayesianOptimizer
        
        mock_evaluator.return_value = None
        
        rag_config = {
            "chunk_size": [256, 512],
            "top_k": [3, 5, 10],
            "temperature": [0.3, 0.7]
        }
        
        optimizer = BayesianOptimizer(
            llm_provider="groq",
            llm_api_key="test-key",
            rag_config=rag_config,
            documents=[]
        )
        
        assert optimizer.chunk_sizes == [256, 512]
        assert optimizer.top_k_values == [3, 5, 10]
    
    @patch('autorag.optimization.bayesian.get_evaluator')
    def test_initializes_empty_pipeline_cache(self, mock_evaluator):
        """Should start with empty pipeline cache."""
        from autorag.optimization.bayesian import BayesianOptimizer
        
        mock_evaluator.return_value = None
        
        optimizer = BayesianOptimizer(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        assert optimizer._pipeline_cache == {}


class TestBayesianResults:
    """Test result handling."""
    
    @patch('autorag.optimization.bayesian.get_evaluator')
    def test_get_best_config_raises_without_results(self, mock_evaluator):
        """get_best_config should raise if optimize() not called."""
        from autorag.optimization.bayesian import BayesianOptimizer
        
        mock_evaluator.return_value = None
        
        optimizer = BayesianOptimizer(
            llm_provider="groq",
            llm_api_key="test-key"
        )
        
        with pytest.raises(ValueError):
            optimizer.get_best_config()


class TestEarlyStopping:
    """Test early stopping callback."""
    
    def test_callback_initializes_correctly(self):
        """EarlyStoppingCallback should initialize with correct values."""
        from autorag.optimization.bayesian import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(patience=5, min_trials=10)
        
        assert callback.patience == 5
        assert callback.min_trials == 10
        assert callback.trials_without_improvement == 0
