"""
Vector store manager for Pinecone.
Handles storing and retrieving document embeddings.
"""
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import time


class VectorStore:
    """Manages Pinecone vector database operations."""
    
    def __init__(self, api_key: str, index_name: str, dimension: int = 384):
        """
        Initialize Pinecone connection.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of Pinecone index (e.g., 'autorag')
            dimension: Embedding dimension (default: 384)
        """
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Connect to or create index
        self._ensure_index_exists()
        self.index = self.pc.Index(index_name)
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create serverless index (free tier)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",  # Cosine similarity for text
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Adjust based on your location
                )
            )
            # Wait for index to be ready
            time.sleep(1)
    
    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store documents with their embeddings in Pinecone.
        
        Args:
            documents: List of document dicts with 'id', 'text', 'metadata'
            embeddings: List of embedding vectors (same order as documents)
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
        
        # Prepare vectors for upsert
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vectors.append({
                "id": str(doc["id"]),
                "values": embedding,
                "metadata": {
                    "text": doc["text"][:1000],  # Limit text size (Pinecone metadata limit)
                    **doc.get("metadata", {})  # Include original metadata
                }
            })
        
        # Batch upsert (Pinecone handles up to 1000 vectors per batch)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar documents to query.
        
        Args:
            query_embedding: Query vector (384-dim)
            top_k: Number of results to return (default: 5)
            
        Returns:
            List of matches with 'id', 'score', 'text', 'metadata'
        """
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": match.score,  # Cosine similarity score
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            })
        
        return matches
    
    def delete_all(self):
        """Delete all vectors from index (useful for resetting)."""
        self.index.delete(delete_all=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics (total vector count, etc.)."""
        return self.index.describe_index_stats()
