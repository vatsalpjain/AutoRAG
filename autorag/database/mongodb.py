"""
MongoDB connector for AutoRAG.
Fetches documents from MongoDB collections.
"""
from typing import List, Dict, Any
from pymongo import MongoClient
from autorag.utils.config import DatabaseConfig


class MongoDBConnector:
    """Connector for MongoDB collections."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize MongoDB connector.
        
        Args:
            config: DatabaseConfig object with MongoDB credentials
        """
        if config.type != "mongodb":
            raise ValueError(f"Invalid database type: {config.type}. Expected 'mongodb'")
        
        self.config = config
        # Connect using the connection string (e.g., mongodb://localhost:27017)
        self.client = MongoClient(config.connection_string)
        # Get reference to database and collection
        self.db = self.client[config.database]
        self.collection = self.db[config.collection]
        # Column/field names for document structure
        self.text_column = config.text_column or "content"
        self.id_column = config.id_column or "_id"
    
    def test_connection(self) -> bool:
        """
        Test connection to MongoDB.
        
        Returns:
            True if connection successful
        """
        try:
            # Ping the server to verify connection
            self.client.admin.command("ping")
            return True
        except Exception as e:
            raise Exception(f"Failed to connect to MongoDB: {e}")
    
    def fetch_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch documents from MongoDB collection.
        Returns documents in standard format for RAG pipeline.
        
        Args:
            limit: Maximum number of documents to fetch (default: 100)
            
        Returns:
            List of documents with 'id', 'text', and 'metadata' keys
        """
        try:
            # Query collection with limit
            cursor = self.collection.find().limit(limit)
            
            documents = []
            for doc in cursor:
                # Get document ID (convert ObjectId to string)
                doc_id = str(doc.get(self.id_column, ""))
                
                # Get text content from configured column
                text_content = doc.get(self.text_column, "")
                
                if not text_content:
                    continue  # Skip documents without text
                
                # Build metadata from all other fields
                metadata = {k: v for k, v in doc.items() 
                           if k not in [self.id_column, self.text_column]}
                # Convert ObjectId in metadata to string for JSON serialization
                metadata = {k: str(v) if hasattr(v, '__str__') and type(v).__name__ == 'ObjectId' 
                           else v for k, v in metadata.items()}
                metadata["source"] = f"mongodb:{self.config.database}.{self.config.collection}"
                metadata["size"] = len(text_content)
                
                documents.append({
                    "id": doc_id,
                    "text": text_content,
                    "metadata": metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to fetch documents from MongoDB: {e}")
    
    def count_documents(self) -> int:
        """Count total documents in collection."""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            raise Exception(f"Failed to count documents: {e}")
