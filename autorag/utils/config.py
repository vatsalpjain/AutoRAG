"""
Configuration loader and validator for AutoRAG.
Uses Pydantic for type-safe config validation.
"""
from pathlib import Path
from typing import Optional, Literal, List
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
    # Storage bucket fields (new - for file-based storage)
    bucket: Optional[str] = Field(default="pdf", description="Supabase Storage bucket name")
    folder: Optional[str] = Field(default="pdf", description="Folder path within bucket")
    # Table fields (legacy - for table-based storage)
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


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider: Literal["groq", "openai", "openrouter"] = Field(
        default="groq",
        description="LLM provider: groq, openai, or openrouter"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name (null = use provider default)"
    )


class APIKeysConfig(BaseModel):
    """API keys for external services."""
    
    # LLM provider keys (at least one required based on llm.provider)
    groq: Optional[str] = Field(default=None, description="Groq API key")
    openai: Optional[str] = Field(default=None, description="OpenAI API key")
    openrouter: Optional[str] = Field(default=None, description="OpenRouter API key")
    # NOTE: Pinecone removed - now using local ChromaDB


class OptimizationConfig(BaseModel):
    """Optimization settings."""
    
    strategy: Literal["grid", "bayesian"] = Field(
        default="grid",
        description="Optimization strategy: 'grid' (exhaustive) or 'bayesian' (Optuna)"
    )
    num_experiments: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of experiments to run (1-100)"
    )
    test_questions: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Number of test questions to generate (5-500)"
    )


class EvaluationConfig(BaseModel):
    """Evaluation settings."""
    
    method: Literal["custom", "ragas"] = Field(
        default="custom",
        description="Evaluation method: 'custom' (built-in) or 'ragas' (official library)"
    )


class RAGConfig(BaseModel):
    """
    RAG parameter search space for optimization.
    Users define lists of values to try for each parameter.
    """
    
    # === INDEXING PARAMETERS (require re-indexing for each combo) ===
    chunk_size: List[int] = Field(
        default=[500],
        description="Chunk sizes to try (characters)"
    )
    chunk_overlap: List[int] = Field(
        default=[50],
        description="Chunk overlaps to try (characters)"
    )
    embedding_model: List[str] = Field(
        default=["all-MiniLM-L6-v2"],
        description="Embedding models to try (HuggingFace names)"
    )
    
    # === QUERY PARAMETERS (fast to test, no re-indexing) ===
    top_k: List[int] = Field(
        default=[3, 5, 10],
        description="Number of documents to retrieve"
    )
    temperature: List[float] = Field(
        default=[0.3, 0.7, 1.0],
        description="LLM temperature values"
    )
    
    @field_validator("chunk_size", "chunk_overlap", "top_k")
    @classmethod
    def validate_positive_ints(cls, v: List[int]) -> List[int]:
        """Ensure all values are positive integers."""
        if not v:
            raise ValueError("At least one value must be specified")
        if any(x <= 0 for x in v):
            raise ValueError("All values must be positive")
        return v
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: List[float]) -> List[float]:
        """Ensure temperature values are valid (0-2)."""
        if not v:
            raise ValueError("At least one temperature value must be specified")
        if any(x < 0 or x > 2 for x in v):
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_models(cls, v: List[str]) -> List[str]:
        """Ensure at least one embedding model is specified."""
        if not v:
            raise ValueError("At least one embedding model must be specified")
        return v


class Config(BaseModel):
    """Main configuration object for AutoRAG."""
    
    database: DatabaseConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api_keys: APIKeysConfig
    optimization: OptimizationConfig
    rag: RAGConfig = Field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    @field_validator("api_keys")
    @classmethod
    def validate_provider_key(cls, v: APIKeysConfig, info: ValidationInfo) -> APIKeysConfig:
        """Ensure API key exists for selected provider."""
        llm_config = info.data.get("llm")
        if llm_config:
            provider = llm_config.provider
            key = getattr(v, provider, None)
            if not key or key.strip() == "":
                raise ValueError(f"{provider} API key is required when llm.provider='{provider}'")
        return v


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
