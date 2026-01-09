"""
Local vector store using ChromaDB.
Stores embeddings locally - no API key required, unlimited usage.
"""
import chromadb
from typing import List, Dict, Any


class ChromaVectorStore:
    """
    Manages local vector storage using ChromaDB.
    Supports multiple collections for different indexing configurations.
    """
    
    def __init__(self, collection_name: str = "autorag", persist_dir: str = ".autorag_cache"):
        """
        Initialize ChromaDB connection.
        
        Args:
            collection_name: Name of the collection (e.g., 'autorag_500_50_minilm')
            persist_dir: Directory to persist the database (default: .autorag_cache)
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize persistent client (data survives restarts)
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store documents with their embeddings.
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
            embeddings: List of embedding vectors (same order as documents)
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
        
        if not documents:
            return
        
        # ChromaDB requires string IDs
        ids = [str(doc["id"]) for doc in documents]
        texts = [doc["text"] for doc in documents]
        # ChromaDB requires non-empty metadata, add placeholder if empty
        metadatas = []
        for doc in documents:
            meta = doc.get("metadata", {})
            if not meta:
                meta = {"_placeholder": "true"}
            metadatas.append(meta)
        
        # Upsert in batches (ChromaDB handles this well, but let's be safe)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar documents to query.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results (default: 5)
            
        Returns:
            List of matches with 'id', 'score', 'text', 'metadata'
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results to match Pinecone interface
        matches = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity score
                # For cosine: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance
                
                matches.append({
                    "id": doc_id,
                    "score": score,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })
        
        return matches
    
    def delete_all(self):
        """Delete all vectors from collection (recreates it)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        """Get number of vectors in collection."""
        return self.collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics (compatible with Pinecone interface)."""
        return {
            "total_vector_count": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir
        }
