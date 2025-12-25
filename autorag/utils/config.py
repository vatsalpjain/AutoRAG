"""
Configuration loader and validator for AutoRAG.
Uses Pydantic for type-safe config validation.
"""
from pathlib import Path
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    type: Literal["supabase", "mongodb", "postgresql"] = Field(
        description="Type of database to connect to"
    )
    
    # Supabase fields
    url: Optional[str] = None
    key: Optional[str] = None
    table: Optional[str] = None
    text_column: Optional[str] = Field(default="content", description="Column containing document text")
    id_column: Optional[str] = Field(default="id", description="Column containing unique identifier")
    
    # MongoDB fields
    connection_string: Optional[str] = None
    database: Optional[str] = None
    collection: Optional[str] = None
    text_column: Optional[str] = Field(default="content", description="Field containing document text")
    id_column: Optional[str] = Field(default="_id", description="Field containing unique identifier")
    
    # PostgreSQL fields
    host: Optional[str] = None
    port: Optional[int] = None
    text_column: Optional[str] = Field(default="content", description="Column containing document text")
    id_column: Optional[str] = Field(default="id", description="Column containing unique identifier")
    user: Optional[str] = None
    password: Optional[str] = None
    
    @field_validator("url")
    @classmethod
    def validate_supabase_url(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate Supabase URL if type is supabase."""
        if info.data.get("type") == "supabase" and not v:
            raise ValueError("Supabase URL is required when type is 'supabase'")
        return v
    
    @field_validator("key")
    @classmethod
    def validate_supabase_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate Supabase key if type is supabase."""
        if info.data.get("type") == "supabase" and not v:
            raise ValueError("Supabase key is required when type is 'supabase'")
        return v
    
    @field_validator("connection_string")
    @classmethod
    def validate_mongodb_connection(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate MongoDB connection string if type is mongodb."""
        if info.data.get("type") == "mongodb" and not v:
            raise ValueError("MongoDB connection string is required when type is 'mongodb'")
        return v
    
    @field_validator("host")
    @classmethod
    def validate_postgres_host(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate PostgreSQL host if type is postgresql."""
        if info.data.get("type") == "postgresql" and not v:
            raise ValueError("PostgreSQL host is required when type is 'postgresql'")
        return v


class APIKeysConfig(BaseModel):
    """API keys for external services."""
    
    groq: str = Field(description="Groq API key (required - default LLM)")
    pinecone: str = Field(description="Pinecone API key")
    pinecone_index: str = Field(default="autorag", description="Pinecone index name")
    
    @field_validator("groq")
    @classmethod
    def validate_groq_key(cls, v: str) -> str:
        """Ensure Groq key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("Groq API key cannot be empty")
        return v
    
    @field_validator("pinecone")
    @classmethod
    def validate_pinecone_key(cls, v: str) -> str:
        """Ensure Pinecone key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("Pinecone API key cannot be empty")
        return v


class OptimizationConfig(BaseModel):
    """Optimization settings."""
    
    num_experiments: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of experiments to run (1-100)"
    )
    test_questions: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Number of test questions to generate (10-500)"
    )


class Config(BaseModel):
    """Main configuration object for AutoRAG."""
    
    database: DatabaseConfig
    api_keys: APIKeysConfig
    optimization: OptimizationConfig


def load_config(config_path: str | Path) -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        yaml.YAMLError: If YAML syntax is invalid
    """
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create a config.yaml file. See config.yaml.example for template."
        )
    
    # Load YAML file
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")
    
    # Validate and create Config object
    try:
        config = Config(**config_data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
