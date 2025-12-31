"""
PostgreSQL connector for AutoRAG.
Fetches documents from PostgreSQL tables.
"""
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from autorag.utils.config import DatabaseConfig


class PostgreSQLConnector:
    """Connector for PostgreSQL tables."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize PostgreSQL connector.
        
        Args:
            config: DatabaseConfig object with PostgreSQL credentials
        """
        if config.type != "postgresql":
            raise ValueError(f"Invalid database type: {config.type}. Expected 'postgresql'")
        
        self.config = config
        # Build connection parameters
        self.conn_params = {
            "host": config.host,
            "port": config.port or 5432,  # Default PostgreSQL port
            "database": config.database,
            "user": config.user,
            "password": config.password
        }
        # Table and column configuration
        self.table = config.table
        self.text_column = config.text_column or "content"
        self.id_column = config.id_column or "id"
        
        # Create connection
        self.conn = psycopg2.connect(**self.conn_params)
    
    def test_connection(self) -> bool:
        """
        Test connection to PostgreSQL.
        
        Returns:
            True if connection successful
        """
        try:
            # Execute simple query to verify connection
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:
            raise Exception(f"Failed to connect to PostgreSQL: {e}")
    
    def fetch_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch documents from PostgreSQL table.
        Returns documents in standard format for RAG pipeline.
        
        Args:
            limit: Maximum number of documents to fetch (default: 100)
            
        Returns:
            List of documents with 'id', 'text', and 'metadata' keys
        """
        try:
            # Use RealDictCursor for dict-like row access
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build and execute query
                query = f'SELECT * FROM "{self.table}" LIMIT %s'
                cur.execute(query, (limit,))
                rows = cur.fetchall()
            
            documents = []
            for row in rows:
                # Convert row to regular dict
                row_dict = dict(row)
                
                # Get document ID
                doc_id = str(row_dict.get(self.id_column, ""))
                
                # Get text content
                text_content = row_dict.get(self.text_column, "")
                
                if not text_content:
                    continue  # Skip rows without text
                
                # Build metadata from all other columns
                metadata = {k: v for k, v in row_dict.items() 
                           if k not in [self.id_column, self.text_column]}
                metadata["source"] = f"postgresql:{self.config.database}.{self.table}"
                metadata["size"] = len(str(text_content))
                
                documents.append({
                    "id": doc_id,
                    "text": str(text_content),
                    "metadata": metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to fetch documents from PostgreSQL: {e}")
    
    def count_documents(self) -> int:
        """Count total rows in table."""
        try:
            with self.conn.cursor() as cur:
                query = f'SELECT COUNT(*) FROM "{self.table}"'
                cur.execute(query)
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            raise Exception(f"Failed to count documents: {e}")
