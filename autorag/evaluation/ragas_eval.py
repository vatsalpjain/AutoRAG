"""
RAGAS Evaluator - Evaluates RAG pipeline using Ragas metrics.
Compatible with Ragas v0.4+ and Groq LLM with round-robin model rotation.
"""
import os
import time
from typing import List, Dict, Any, Optional, ClassVar
from datasets import Dataset
from groq import Groq, RateLimitError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_similarity,
)


class ChatGroqRoundRobin(BaseChatModel):
    """
    LangChain-compatible Groq chat wrapper with round-robin model rotation.
    Rotates between multiple models to bypass rate limits (30 RPM per model).
    """
    
    # Class-level model rotation (shared across all instances for true round-robin)
    MODELS: ClassVar[List[str]] = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    _current_model_index: ClassVar[int] = 0
    _model_stats: ClassVar[Dict[str, Dict[str, Any]]] = {}
    
    # Instance fields for Pydantic
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 5
    base_delay: float = 1.0
    
    # Store the Groq client (allow arbitrary types)
    model_config = {"arbitrary_types_allowed": True}
    _client: Optional[Groq] = None
    
    def __init__(self, api_key: str, temperature: float = 0.0, max_tokens: int = 1024, **kwargs):
        """Initialize ChatGroq with round-robin rotation."""
        super().__init__(
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        # Initialize Groq client after super().__init__
        object.__setattr__(self, '_client', Groq(api_key=api_key))
        
        # Initialize class-level stats if empty
        if not ChatGroqRoundRobin._model_stats:
            ChatGroqRoundRobin._model_stats = {
                model: {"requests": 0, "failures": 0, "last_used": None}
                for model in self.MODELS
            }
    
    @classmethod
    def _get_current_model(cls) -> str:
        """Get the current model in rotation."""
        return cls.MODELS[cls._current_model_index]
    
    @classmethod
    def _rotate_model(cls):
        """Rotate to the next model in the list."""
        cls._current_model_index = (cls._current_model_index + 1) % len(cls.MODELS)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> ChatResult:
        """
        Generate a response with automatic model rotation and retry.
        This is the core method called by LangChain/Ragas.
        """
        # Convert LangChain messages to Groq format
        groq_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                groq_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                groq_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                groq_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Fallback for other message types
                groq_messages.append({"role": "user", "content": str(msg.content)})
        
        # Get kwargs or use defaults
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Retry loop with model rotation
        last_error = None
        for attempt in range(self.max_retries):
            # Get current model from rotation
            current_model = self._get_current_model()
            
            try:
                # Call Groq API
                response = self._client.chat.completions.create(
                    model=current_model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Update success stats
                ChatGroqRoundRobin._model_stats[current_model]["requests"] += 1
                ChatGroqRoundRobin._model_stats[current_model]["last_used"] = time.time()
                
                # Rotate to next model for true round-robin
                self._rotate_model()
                
                # Extract content and return LangChain format
                content = response.choices[0].message.content
                generation = ChatGeneration(message=AIMessage(content=content))
                return ChatResult(generations=[generation])
                
            except RateLimitError as e:
                # Record failure
                ChatGroqRoundRobin._model_stats[current_model]["failures"] += 1
                last_error = e
                
                # Rotate to next model for immediate retry
                self._rotate_model()
                
                # Calculate exponential backoff delay
                delay = self.base_delay * (2 ** attempt)
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                last_error = e
                # For non-rate-limit errors, rotate and retry once
                if attempt == 0:
                    self._rotate_model()
                    time.sleep(0.5)
                    continue
                else:
                    raise e
        
        # If we exhausted all retries, raise the last error
        raise last_error or Exception("Failed after max retries")
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "chat_groq_round_robin"
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get current rotation statistics."""
        return {
            "current_model": cls._get_current_model(),
            "model_stats": cls._model_stats.copy()
        }
    
    @classmethod
    def reset_rotation(cls):
        """Reset model rotation to first model."""
        cls._current_model_index = 0


class RagasEvaluator:
    """Evaluates RAG outputs using Ragas metrics with Groq LLM (round-robin rotation)."""

    def __init__(self, groq_api_key: str):
        """Initialize Ragas evaluator with round-robin Groq chat model."""
        os.environ["GROQ_API_KEY"] = groq_api_key

        # Use round-robin ChatGroq for rate limit handling
        chat_groq = ChatGroqRoundRobin(
            api_key=groq_api_key,
            temperature=0.0,
            max_tokens=1024,
        )
        self.llm = LangchainLLMWrapper(chat_groq)
        self.chat_model = chat_groq  # Keep reference for stats

        # Use HuggingFace embeddings since Groq lacks embeddings API
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    def prepare_dataset(self, qa_pairs, rag_results):
        """
        Prepare dataset for RAGAS evaluation.
        
        Args:
            qa_pairs: List of Q&A pairs with 'question', 'answer', 'document_id', 'metadata'
            rag_results: List of RAG results with 'question', 'answer', 'retrieved_docs'
            
        Returns:
            HuggingFace Dataset object (required by Ragas v0.4+)
        """
        # Ragas v0.4+ uses different field names
        dataset_dict = {
            'user_input': [],          # Changed from 'question'
            'response': [],            # Changed from 'answer' (generated by RAG)
            'retrieved_contexts': [],  # Changed from 'contexts'
            'reference': []            # Changed from 'ground_truth'
        }
        
        # Create lookup dict for efficient matching
        rag_dict = {item['question']: item for item in rag_results}
        
        # Build dataset
        for qa in qa_pairs:
            question = qa['question']
            if question in rag_dict:
                rag_item = rag_dict[question]
                
                dataset_dict['user_input'].append(question)
                dataset_dict['reference'].append(qa['answer'])
                dataset_dict['response'].append(rag_item['answer'])
                
                # Extract text from retrieved docs (must be list of strings)
                contexts = [doc['text'] for doc in rag_item['retrieved_docs']]
                dataset_dict['retrieved_contexts'].append(contexts)
        
        # Convert to HuggingFace Dataset (required by Ragas v0.4+)
        return Dataset.from_dict(dataset_dict)
    
    def evaluate(self, dataset):
        """
        Evaluate RAG outputs using RAGAS metrics.
        
        Args:
            dataset: HuggingFace Dataset object
            
        Returns:
            Dict of RAGAS metric scores (averaged across all rows)
        """
        # Define metrics to evaluate (reduced to 3 most important for faster evaluation)
        metrics_list = [
            answer_relevancy,      # 30% - Most important
            faithfulness,          # 25% - Hallucination detection
            answer_similarity      # 15% - Quality check
            # Removed: context_precision, context_recall (less critical, saves ~40% of calls)
        ]
        
        # Call Ragas evaluate with LLM, embeddings, and metrics
        evaluation_result = evaluate(
            dataset=dataset,
            llm=self.llm,
            embeddings=self.embeddings,  # Required by Ragas v0.4+
            metrics=metrics_list
        )
        
        # Ragas v0.4+ returns EvaluationResult with scores as List[Dict]
        # Each dict contains scores for one row, we need to average them
        scores_list = evaluation_result.scores  # List of dicts
        
        if not scores_list:
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "answer_similarity": 0.0,
            }
        
        # Get all metric names from first row
        metric_names = [k for k in scores_list[0].keys() if k not in ['user_input', 'response', 'retrieved_contexts', 'reference']]
        
        # Calculate average for each metric
        results = {}
        for metric in metric_names:
            values = [row.get(metric, 0.0) for row in scores_list if row.get(metric) is not None]
            results[metric] = sum(values) / len(values) if values else 0.0
        
        return results
    
    def calculate_aggregate_score(self, evaluation_results):
        """
        Calculate aggregate RAGAS score from individual metric scores.
        
        Args:
            evaluation_results: Dict of RAGAS metric scores
            
        Returns:
            Aggregate RAGAS score (float)
        """
        # Updated weights for 3-metric evaluation (normalized to sum to 1.0)
        weights = {
            "answer_relevancy": 0.45,     # Was 0.30
            "faithfulness": 0.35,         # Was 0.25
            "answer_similarity": 0.20     # Was 0.15
        }
        
        aggregate_score = 0.0
        for metric, weight in weights.items():
            score = evaluation_results.get(metric, 0.0)
            aggregate_score += score * weight
            
        return aggregate_score
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get round-robin model usage statistics."""
        return ChatGroqRoundRobin.get_stats()
       
