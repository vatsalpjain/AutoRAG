"""
Shared test fixtures for AutoRAG tests.
"""
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms.",
            "metadata": {"source": "test"}
        },
        {
            "id": "doc2", 
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            "metadata": {"source": "test"}
        },
        {
            "id": "doc3",
            "text": "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models to provide more accurate and grounded responses.",
            "metadata": {"source": "test"}
        }
    ]


@pytest.fixture
def sample_qa_pairs():
    """Sample Q&A pairs for testing."""
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language known for simplicity and readability."
        },
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables systems to learn from data."
        }
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_rag_config():
    """Sample RAG configuration for testing."""
    return {
        "chunk_size": [500],
        "chunk_overlap": [50],
        "embedding_model": ["all-MiniLM-L6-v2"],
        "top_k": [3, 5],
        "temperature": [0.5]
    }
