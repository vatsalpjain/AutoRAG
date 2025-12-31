"""
Groq Multi-Model Client - Rotates between models to bypass rate limits.
Handles 429 errors with automatic fallback and exponential backoff retry.
"""
import time
from typing import List, Dict, Any, Optional
from groq import Groq
from groq import RateLimitError


class GroqMultiModelClient:
    """
    Groq client that rotates between multiple models to avoid rate limits.
    Compatible interface with standard Groq client.
    """
    
    # Model rotation list (30 RPM per model)
    MODELS = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    
    def __init__(self, api_key: str):
        """
        Initialize multi-model Groq client.
        
        Args:
            api_key: Groq API key
        """
        self.client = Groq(api_key=api_key)
        self.current_model_index = 0
        
        # Track usage per model
        self.model_stats = {
            model: {
                "requests": 0,
                "failures": 0,
                "last_used": None
            }
            for model in self.MODELS
        }
    
    @property
    def chat(self):
        """Provide .chat property for compatibility with Groq client."""
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
        Create a chat completion with automatic model rotation and retry.
        Compatible with groq.chat.completions.create() interface.
        
        Args:
            model: Model name (ignored - uses rotation)
            messages: Chat messages
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to Groq API
            
        Returns:
            Groq ChatCompletion response
        """
        max_retries = 5
        base_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            # Get current model
            current_model = self._get_next_model()
            
            try:
                # Call Groq API
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                # Update success stats
                self.model_stats[current_model]["requests"] += 1
                self.model_stats[current_model]["last_used"] = time.time()
                
                # Rotate to next model for true round-robin
                self._rotate_model()
                
                # Proactive delay to prevent rate limits (1 sec between calls)
                time.sleep(1.0)
                
                return response
                
            except RateLimitError as e:
                # Record failure
                self.model_stats[current_model]["failures"] += 1
                
                # Rotate to next model for immediate retry
                self._rotate_model()
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                
                # If this is the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise RateLimitError(
                        f"Rate limit exceeded on all models after {max_retries} attempts. "
                        f"Model stats: {self._format_stats()}"
                    )
                
                # Wait before retry (exponential backoff)
                time.sleep(delay)
                
            except Exception as e:
                # For non-rate-limit errors, rotate and retry once
                if attempt == 0:
                    self._rotate_model()
                    time.sleep(1)
                    continue
                else:
                    # Re-raise after one retry
                    raise e
    
    def _get_next_model(self) -> str:
        """Get the current model in rotation."""
        return self.MODELS[self.current_model_index]
    
    def _rotate_model(self):
        """Rotate to the next model in the list."""
        self.current_model_index = (self.current_model_index + 1) % len(self.MODELS)
    
    def get_current_limits(self) -> Dict[str, Any]:
        """
        Get current rate limit status for each model.
        
        Returns:
            Dict with model statistics
        """
        return {
            "current_model": self.MODELS[self.current_model_index],
            "model_stats": self.model_stats.copy()
        }
    
    def reset_rotation(self):
        """Reset model rotation to first model."""
        self.current_model_index = 0
    
    def _format_stats(self) -> str:
        """Format model statistics for error messages."""
        stats_str = []
        for model, stats in self.model_stats.items():
            stats_str.append(
                f"{model}: {stats['requests']} requests, {stats['failures']} failures"
            )
        return " | ".join(stats_str)
