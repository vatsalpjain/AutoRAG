"""
Supabase Storage connector for AutoRAG.
Fetches raw documents from Supabase Storage bucket.
"""
from typing import List, Dict, Any
from supabase import create_client, Client
from autorag.utils.config import DatabaseConfig


class SupabaseConnector:
    """Connector for Supabase Storage bucket."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize Supabase connector.
        
        Args:
            config: DatabaseConfig object with Supabase credentials
        """
        if config.type != "supabase":
            raise ValueError(f"Invalid database type: {config.type}. Expected 'supabase'")
        
        self.config = config
        self.client: Client = create_client(config.url, config.key)
        self.bucket_name = config.bucket or "pdf"
        self.folder = config.folder or "pdf"
    
    def test_connection(self) -> bool:
        """
        Test connection to Supabase Storage.
        
        Returns:
            True if connection successful
        """
        try:
            self.client.storage.from_(self.bucket_name).list(self.folder)
            return True
        except Exception as e:
            raise Exception(f"Failed to connect to Supabase Storage: {e}")
    
    def fetch_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch raw documents from Supabase Storage bucket.
        Returns full text content - chunking handled by RAG pipeline.
        
        Args:
            limit: Maximum number of files to fetch (default: 100)
            
        Returns:
            List of documents with 'id', 'text', and 'metadata' keys
        """
        try:
            # List files in the folder
            files = self.client.storage.from_(self.bucket_name).list(self.folder)
            
            if not files:
                return []
            
            # Filter for text files only
            text_files = [f for f in files if f.get("name", "").endswith(".txt")][:limit]
            
            documents = []
            for file_info in text_files:
                file_name = file_info.get("name")
                if not file_name:
                    continue
                
                file_path = f"{self.folder}/{file_name}"
                
                try:
                    file_bytes = self.client.storage.from_(self.bucket_name).download(file_path)
                    text_content = file_bytes.decode("utf-8")
                    
                    documents.append({
                        "id": file_name,
                        "text": text_content,
                        "metadata": {
                            "source": file_name,
                            "size": len(text_content),
                            "bucket": self.bucket_name,
                            "path": file_path
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to fetch documents from Supabase Storage: {e}")
    
    def count_documents(self) -> int:
        """Count total text files in folder."""
        try:
            files = self.client.storage.from_(self.bucket_name).list(self.folder)
            text_files = [f for f in files if f.get("name", "").endswith(".txt")]
            return len(text_files)
        except Exception as e:
            raise Exception(f"Failed to count documents: {e}")
