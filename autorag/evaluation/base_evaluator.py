"""
Base Evaluator - Abstract interface for RAG evaluation methods.
All evaluators (Custom, RAGAS) must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseEvaluator(ABC):
    """
    Abstract base class for RAG evaluators.
    Defines the common interface that all evaluation methods must implement.
    """
    
    @abstractmethod
    def get_evaluation_method(self) -> str:
        """
        Return the evaluation method identifier.
        
        Returns:
            String identifier: "custom" or "ragas"
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        reference: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single Q&A pair on all metrics.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context (concatenated)
            reference: Ground truth answer (optional)
            
        Returns:
            Dict with metric scores (answer_relevancy, faithfulness, etc.)
        """
        pass
    
    @abstractmethod
    def calculate_aggregate_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted aggregate score from individual metrics.
        
        Args:
            scores: Dict of individual metric scores
            
        Returns:
            Aggregate score 0.0 to 1.0
        """
        pass
