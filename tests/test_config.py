"""
Tests for RAGConfig validation in autorag.utils.config
"""
import pytest
from pydantic import ValidationError

from autorag.utils.config import RAGConfig


class TestRAGConfigDefaults:
    """Test RAGConfig default values."""
    
    def test_default_values(self):
        """RAGConfig should have sensible defaults."""
        config = RAGConfig()
        assert config.chunk_size == [500]
        assert config.chunk_overlap == [50]
        assert config.embedding_model == ["all-MiniLM-L6-v2"]
        assert config.top_k == [3, 5, 10]
        assert config.temperature == [0.3, 0.7, 1.0]
    
    def test_custom_values(self):
        """RAGConfig should accept custom values."""
        config = RAGConfig(
            chunk_size=[256, 512],
            top_k=[5],
            temperature=[0.5]
        )
        assert config.chunk_size == [256, 512]
        assert config.top_k == [5]
        assert config.temperature == [0.5]


class TestRAGConfigValidation:
    """Test RAGConfig validation rules."""
    
    def test_rejects_empty_chunk_size(self):
        """Should reject empty chunk_size list."""
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=[])
    
    def test_rejects_negative_chunk_size(self):
        """Should reject negative chunk_size values."""
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=[-100])
    
    def test_rejects_zero_chunk_size(self):
        """Should reject zero chunk_size."""
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=[0])
    
    def test_rejects_empty_top_k(self):
        """Should reject empty top_k list."""
        with pytest.raises(ValidationError):
            RAGConfig(top_k=[])
    
    def test_rejects_negative_top_k(self):
        """Should reject negative top_k values."""
        with pytest.raises(ValidationError):
            RAGConfig(top_k=[-5])
    
    def test_rejects_temperature_too_high(self):
        """Should reject temperature > 2."""
        with pytest.raises(ValidationError):
            RAGConfig(temperature=[3.0])
    
    def test_rejects_temperature_negative(self):
        """Should reject negative temperature."""
        with pytest.raises(ValidationError):
            RAGConfig(temperature=[-0.5])
    
    def test_rejects_empty_embedding_model(self):
        """Should reject empty embedding_model list."""
        with pytest.raises(ValidationError):
            RAGConfig(embedding_model=[])
    
    def test_accepts_valid_temperature_range(self):
        """Should accept temperature in valid range [0, 2]."""
        config = RAGConfig(temperature=[0.0, 1.0, 2.0])
        assert config.temperature == [0.0, 1.0, 2.0]
