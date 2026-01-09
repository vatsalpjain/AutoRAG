"""
Embedding service using HuggingFace sentence-transformers.
Supports multiple models with automatic dimension detection.
"""
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingService:
    """Handles text embeddings using HuggingFace models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        # Load model (downloads on first run, then cached locally)
        self.model = SentenceTransformer(model_name)
        # Auto-detect dimension from model (works for any model)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
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
        """Return embedding dimension (auto-detected from model)."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Return the model name being used."""
        return self.model_name
