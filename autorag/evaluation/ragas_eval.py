"""
RAGAS Evaluator - Wrapper for official RAGAS library.
Provides the same interface as CustomEvaluator but uses RAGAS metrics.
"""
import warnings
from typing import Dict, Any, List, Optional

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RagasEvaluator:
    """
    RAGAS library wrapper implementing the same interface as CustomEvaluator.
    Uses official RAGAS metrics for evaluation.
    """
    
    # Metric weights for RAGAS aggregate score
    WEIGHTS = {
        "answer_relevancy": 0.30,
        "faithfulness": 0.30,
        "answer_similarity": 0.25,
        "context_recall": 0.15
    }
    
    def __init__(self, llm_provider: str, llm_api_key: str, llm_model: str = None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm_provider: LLM provider (groq, openai, openrouter)
            llm_api_key: API key for the LLM provider
            llm_model: Optional model name (uses provider default if None)
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS library not installed. Install with: pip install ragas"
            )
        
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        
        # Configure RAGAS to use OpenAI-compatible API
        self._configure_llm()
    
    def _configure_llm(self):
        """Configure RAGAS to use the specified LLM provider."""
        import os
        
        # RAGAS uses OpenAI by default, so we set the API key
        # For Groq/OpenRouter, we need to set the base URL as well
        if self.llm_provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.llm_api_key
        elif self.llm_provider == "groq":
            os.environ["OPENAI_API_KEY"] = self.llm_api_key
            os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
        elif self.llm_provider == "openrouter":
            os.environ["OPENAI_API_KEY"] = self.llm_api_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    
    def get_evaluation_method(self) -> str:
        """Return the evaluation method identifier."""
        return "ragas"
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        reference: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single Q&A pair using RAGAS metrics.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context (concatenated)
            reference: Ground truth answer (optional)
            
        Returns:
            Dict with all metric scores
        """
        # Prepare data for RAGAS
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[context]],  # RAGAS expects list of contexts
        }
        
        if reference:
            data["ground_truth"] = [reference]
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on available data
        metrics = [answer_relevancy, faithfulness]
        if reference:
            metrics.extend([answer_similarity, context_recall])
        
        try:
            # Run RAGAS evaluation
            result = evaluate(dataset, metrics=metrics)
            
            scores = {
                "answer_relevancy": result.get("answer_relevancy", 0.5),
                "faithfulness": result.get("faithfulness", 0.5),
            }
            
            if reference:
                scores["answer_similarity"] = result.get("answer_similarity", 0.5)
                scores["context_recall"] = result.get("context_recall", 0.5)
            
            return scores
            
        except Exception as e:
            warnings.warn(f"RAGAS evaluation failed: {e}. Returning default scores.")
            scores = {"answer_relevancy": 0.5, "faithfulness": 0.5}
            if reference:
                scores["answer_similarity"] = 0.5
                scores["context_recall"] = 0.5
            return scores
    
    def calculate_aggregate_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted aggregate score from individual metrics.
        
        Weights:
        - answer_relevancy: 0.30
        - faithfulness: 0.30
        - answer_similarity: 0.25
        - context_recall: 0.15
        
        Returns:
            Aggregate score 0.0 to 1.0
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, weight in self.WEIGHTS.items():
            if metric in scores and scores[metric] is not None:
                weighted_sum += scores[metric] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Fallback
        
        return weighted_sum / total_weight
    
    def evaluate_batch(
        self,
        qa_pairs: List[Dict[str, Any]],
        rag_results: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple Q&A pairs.
        
        Args:
            qa_pairs: List of {"question": str, "answer": str (ground truth)}
            rag_results: List of {"question": str, "answer": str, "retrieved_docs": [...]}
            
        Returns:
            List of score dicts for each pair
        """
        all_scores = []
        
        # Create lookup
        rag_dict = {r["question"]: r for r in rag_results}
        
        for qa in qa_pairs:
            question = qa["question"]
            reference = qa["answer"]
            
            if question in rag_dict:
                rag = rag_dict[question]
                answer = rag["answer"]
                
                # Concatenate contexts
                context = " ".join([doc["text"] for doc in rag.get("retrieved_docs", [])])
                
                scores = self.evaluate(question, answer, context, reference)
                all_scores.append(scores)
            else:
                # No RAG result for this question
                all_scores.append({
                    "answer_relevancy": 0.0,
                    "faithfulness": 0.0,
                    "answer_similarity": 0.0,
                    "context_recall": 0.0
                })
        
        return all_scores
