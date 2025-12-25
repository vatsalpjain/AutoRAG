"""
Embedding service using HuggingFace sentence-transformers.
Converts text to 384-dimensional vectors.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingService:
    """Handles text embeddings using HuggingFace models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
                        Produces 384-dim vectors, fast and efficient
        """
        # Load model (downloads on first run, then cached locally)
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Output dimension for all-MiniLM-L6-v2
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert single text to embedding vector.
        
        Args:
            text: Input text string
            
        Returns:
            List of 384 floats (embedding vector)
        """
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Convert multiple texts to embeddings efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once (default: 32)
            
        Returns:
            List of embedding vectors (each is 384 floats)
        """
        # Batch encoding is faster than encoding one by one
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10  # Show progress for large batches
        )
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Return embedding dimension (384)."""
        return self.dimension
