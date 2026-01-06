"""
Evaluator Factory - Get the appropriate evaluator based on method selection.
Supports 'custom' (built-in) and 'ragas' (official library) methods.
"""
import warnings
from typing import Literal


def get_evaluator(
    method: Literal["custom", "ragas"],
    llm_provider: str,
    llm_api_key: str,
    llm_model: str = None
):
    """
    Factory function to get the appropriate evaluator.
    
    Args:
        method: Evaluation method - "custom" or "ragas"
        llm_provider: LLM provider (groq, openai, openrouter)
        llm_api_key: API key for the LLM provider
        llm_model: Optional model name
        
    Returns:
        Evaluator instance (CustomEvaluator or RagasEvaluator)
        
    Note:
        If 'ragas' is selected but not installed, falls back to 'custom' with a warning.
    """
    if method == "ragas":
        try:
            from autorag.evaluation.ragas_eval import RagasEvaluator
            return RagasEvaluator(
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model=llm_model
            )
        except ImportError:
            warnings.warn(
                "RAGAS library not installed. Falling back to custom evaluation. "
                "Install RAGAS with: pip install ragas"
            )
            # Fall through to custom evaluator
    
    # Default: custom evaluator
    from autorag.evaluation.custom_eval import CustomEvaluator
    return CustomEvaluator(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model
    )
