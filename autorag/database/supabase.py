"""
Supabase database connector for AutoRAG.
Fetches documents from Supabase tables.
"""
from typing import List, Dict, Any
from supabase import create_client, Client
from autorag.utils.config import DatabaseConfig


class SupabaseConnector:
    """Connector for Supabase database."""
    
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
        self.table_name = config.table or "documents"
        self.text_column = config.text_column or "content"
        self.id_column = config.id_column or "id"
    
    def test_connection(self) -> bool:
        """
        Test connection to Supabase.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            Exception: If connection fails
        """
        try:
            # Try to fetch one row to verify connection
            result = self.client.table(self.table_name).select("*").limit(1).execute()
            return True
        except Exception as e:
            raise Exception(f"Failed to connect to Supabase: {e}")
    
    def fetch_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch documents from Supabase table.
        
        Args:
            limit: Maximum number of documents to fetch (default: 100)
            
        Returns:
            List of documents with 'id', 'text', and 'metadata' keys
            
        Example:
            [
                {
                    "id": "123",
                    "text": "This is document content...",
                    "metadata": {"title": "...", "source": "..."}
                }
            ]
        """
        try:
            # Fetch data from Supabase
            result = self.client.table(self.table_name).select("*").limit(limit).execute()
            
            if not result.data:
                return []
            
            # Transform to standard format
            documents = []
            for row in result.data:
                # Extract required fields
                doc_id = row.get(self.id_column)
                text = row.get(self.text_column)
                
                # Skip if missing required fields
                if not doc_id or not text:
                    continue
                
                # Create metadata from remaining fields
                metadata = {k: v for k, v in row.items() 
                           if k not in [self.id_column, self.text_column]}
                
                documents.append({
                    "id": str(doc_id),
                    "text": str(text),
                    "metadata": metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to fetch documents from Supabase: {e}")
    
    def count_documents(self) -> int:
        """
        Count total documents in table.
        
        Returns:
            Number of documents in the table
        """
        try:
            result = self.client.table(self.table_name).select("*", count="exact").execute()
            return result.count or 0
        except Exception as e:
            raise Exception(f"Failed to count documents: {e}")
