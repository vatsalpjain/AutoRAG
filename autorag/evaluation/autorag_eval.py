"""
AutoRAG Evaluator - Custom evaluation mimicking RAGAS formulas.
Token-optimized for Groq with simple text parsing (no JSON mode).
"""
import numpy as np
from typing import List, Dict, Any, Optional
from autorag.rag.groq_client import GroqMultiModelClient
from autorag.rag.embeddings import EmbeddingService


class AutoRAGEvaluator:
    """
    Custom RAG evaluator implementing RAGAS-like metrics.
    Designed for Groq compatibility with token optimization.
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize evaluator with Groq client and embeddings.
        
        Args:
            groq_api_key: Groq API key for LLM calls
        """
        self.llm = GroqMultiModelClient(api_key=groq_api_key)
        self.embeddings = EmbeddingService()
    
    # ========== METRIC 1: ANSWER SIMILARITY (Embedding-based) ==========
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text (e.g., generated answer)
            text2: Second text (e.g., ground truth)
            
        Returns:
            Similarity score 0.0 to 1.0
        """
        emb1 = np.array(self.embeddings.embed_text(text1))
        emb2 = np.array(self.embeddings.embed_text(text2))
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    # ========== METRIC 2: FAITHFULNESS (LLM-based) ==========
    def compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Compute faithfulness score: what fraction of answer claims are supported by context.
        Formula: Supported Claims / Total Claims
        
        Args:
            answer: Generated answer to evaluate
            context: Retrieved context that should support the answer
            
        Returns:
            Faithfulness score 0.0 to 1.0
        """
        # Step 1: Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return 1.0  # No claims = vacuously true
        
        # Step 2: Verify each claim against context (batched for token efficiency)
        verified = self._verify_claims_batch(claims, context)
        
        # Step 3: Calculate ratio
        supported_count = sum(verified)
        return supported_count / len(claims)
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract atomic claims from text using LLM."""
        prompt = f"""List each factual claim in the following text. One claim per line. Be concise.

Text: {text}

Claims:"""
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200  # Token optimized
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse: split by newlines, filter empty
        claims = [line.strip().lstrip("•-123456789. ") for line in content.split("\n")]
        claims = [c for c in claims if c and len(c) > 5]  # Filter noise
        
        return claims
    
    def _verify_claims_batch(self, claims: List[str], context: str) -> List[int]:
        """
        Verify multiple claims in a single LLM call (token efficient).
        Returns list of 1 (supported) or 0 (not supported).
        """
        if not claims:
            return []
        
        # Build batched verification prompt
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
        
        prompt = f"""Context: {context}

For each claim below, respond with 1 if supported by context, 0 if not. One number per line.

Claims:
{claims_text}

Answers (1 or 0 only):"""
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=len(claims) * 3  # ~3 tokens per answer (number + newline)
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse: extract 1s and 0s
        results = []
        for char in content:
            if char == '1':
                results.append(1)
            elif char == '0':
                results.append(0)
        
        # Pad if LLM returned fewer than expected
        while len(results) < len(claims):
            results.append(0)  # Assume not supported if unclear
        
        return results[:len(claims)]
    
    # ========== METRIC 3: ANSWER RELEVANCY (Embedding + LLM) ==========
    def compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Compute answer relevancy using reverse question generation.
        Formula: Avg cosine_similarity(original_question, generated_questions)
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Relevancy score 0.0 to 1.0
        """
        # Step 1: Generate 3 questions this answer could respond to
        generated_questions = self._generate_questions(answer, n=3)
        
        if not generated_questions:
            return 0.5  # Fallback if generation fails
        
        # Step 2: Embed original question
        orig_emb = np.array(self.embeddings.embed_text(question))
        
        # Step 3: Embed generated questions and compute similarities
        similarities = []
        for gen_q in generated_questions:
            gen_emb = np.array(self.embeddings.embed_text(gen_q))
            
            dot = np.dot(orig_emb, gen_emb)
            norm1 = np.linalg.norm(orig_emb)
            norm2 = np.linalg.norm(gen_emb)
            
            if norm1 > 0 and norm2 > 0:
                similarities.append(dot / (norm1 * norm2))
        
        if not similarities:
            return 0.5
        
        return float(np.mean(similarities))
    
    def _generate_questions(self, answer: str, n: int = 3) -> List[str]:
        """Generate N questions that this answer could be responding to."""
        prompt = f"""Generate {n} different questions that the following answer could be responding to. One question per line.

Answer: {answer}

Questions:"""
        
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slight variation for diversity
            max_tokens=150  # Token optimized
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse: split by newlines
        questions = [line.strip().lstrip("•-123456789. ") for line in content.split("\n")]
        questions = [q for q in questions if q and q.endswith("?")]
        
        return questions[:n]
    
    # ========== METRIC 4: CONTEXT RECALL (LLM-based) ==========
    def compute_context_recall(self, reference: str, context: str) -> float:
        """
        Compute context recall: fraction of reference claims supported by context.
        Formula: Supported Reference Claims / Total Reference Claims
        
        Args:
            reference: Ground truth / ideal answer
            context: Retrieved context
            
        Returns:
            Context recall score 0.0 to 1.0
        """
        # Step 1: Extract claims from reference answer
        claims = self._extract_claims(reference)
        
        if not claims:
            return 1.0  # No claims = vacuously true
        
        # Step 2: Verify each claim against context
        verified = self._verify_claims_batch(claims, context)
        
        # Step 3: Calculate ratio
        supported_count = sum(verified)
        return supported_count / len(claims)
    
    # ========== AGGREGATE SCORING ==========
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
            Dict with all metric scores
        """
        scores = {}
        
        # Always compute these
        scores["answer_relevancy"] = self.compute_answer_relevancy(question, answer)
        scores["faithfulness"] = self.compute_faithfulness(answer, context)
        
        # Compute if reference available
        if reference:
            scores["answer_similarity"] = self.compute_similarity(answer, reference)
            scores["context_recall"] = self.compute_context_recall(reference, context)
        
        return scores
    
    def calculate_aggregate_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted aggregate score from individual metrics.
        
        Weights (normalized to 1.0):
        - answer_relevancy: 0.30
        - faithfulness: 0.30
        - answer_similarity: 0.25
        - context_recall: 0.15
        
        Returns:
            Aggregate score 0.0 to 1.0
        """
        weights = {
            "answer_relevancy": 0.30,
            "faithfulness": 0.30,
            "answer_similarity": 0.25,
            "context_recall": 0.15
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, weight in weights.items():
            if metric in scores and scores[metric] is not None:
                weighted_sum += scores[metric] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Fallback
        
        return weighted_sum / total_weight
    
    # ========== BATCH EVALUATION ==========
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
