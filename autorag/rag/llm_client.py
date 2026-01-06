"""
Unified LLM Client - Supports Groq, OpenAI, and OpenRouter providers.
All providers use OpenAI-compatible interface.
"""
import time
from typing import List, Dict, Any, Optional
from groq import Groq


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    Compatible interface for all providers via .chat.completions.create()
    """
    
    # Default models for each provider
    DEFAULTS = {
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-mini",
        "openrouter": "meta-llama/llama-3.3-70b-instruct"
    }
    
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: Provider name (groq, openai, openrouter)
            api_key: API key for the provider
            model: Model name (optional, uses provider default if None)
        """
        self.provider = provider
        self.model = model or self.DEFAULTS.get(provider, "llama-3.3-70b-versatile")
        
        # Initialize appropriate client based on provider
        if provider == "groq":
            self.client = Groq(api_key=api_key)
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif provider == "openrouter":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'groq', 'openai', or 'openrouter'")
    
    @property
    def chat(self):
        """Provide .chat property for compatibility."""
        return self
    
    @property
    def completions(self):
        """Provide .completions property for compatibility."""
        return self
    
    def create(
        self,
        model: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with retry on rate limit.
        Compatible with groq/openai.chat.completions.create() interface.
        
        Args:
            model: Model name (optional, uses instance model if not provided)
            messages: Chat messages
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to API
            
        Returns:
            ChatCompletion response
        """
        max_retries = 5
        base_delay = 1.0
        
        # Use provided model or instance default
        use_model = model if model else self.model
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response
                
            except Exception as e:
                # Handle rate limits with exponential backoff
                if "rate" in str(e).lower() or "429" in str(e):
                    delay = base_delay * (2 ** attempt)
                    if attempt == max_retries - 1:
                        raise Exception(f"Rate limit exceeded after {max_retries} attempts: {e}")
                    time.sleep(delay)
                elif attempt == 0:
                    # Retry once for other errors
                    time.sleep(1)
                    continue
                else:
                    raise e
