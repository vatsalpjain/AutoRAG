"""
Synthetic Q&A Generator - Creates test questions and answers from documents.
Uses Groq LLM to generate realistic questions that test RAG retrieval quality.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from autorag.rag.groq_client import GroqMultiModelClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


class SyntheticQAGenerator:
    """Generates synthetic question-answer pairs from documents using LLM."""
    
    def __init__(
        self,
        groq_api_key: str,
        questions_per_doc: int = 2,
        temperature: float = 0.8
    ):
        """
        Initialize the synthetic Q&A generator.
        
        Args:
            groq_api_key: Groq API key for LLM access
            questions_per_doc: Number of questions to generate per document (default: 2)
            temperature: LLM temperature for creativity (0.0-1.0, default: 0.8)
        """
        self.groq_client = GroqMultiModelClient(api_key=groq_api_key)
        self.questions_per_doc = questions_per_doc
        self.temperature = temperature
        
        # Track statistics
        self.stats = {
            "total_documents": 0,
            "total_questions": 0,
            "failed_generations": 0,
            "invalid_qa_pairs": 0
        }
    
    def generate(
        self,
        documents: List[Dict[str, Any]],
        target_count: int = 50,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic Q&A pairs from documents.
        
        Args:
            documents: List of document dicts with 'id', 'text', 'metadata'
            target_count: Target number of Q&A pairs to generate (default: 50)
            show_progress: Whether to show progress bar (default: True)
            
        Returns:
            List of Q&A pairs: [{"question": str, "answer": str, "document_id": str, "metadata": dict}]
        """
        qa_pairs = []
        self.stats["total_documents"] = len(documents)
        
        # Calculate how many docs we need to process to reach target
        docs_needed = min(len(documents), (target_count // self.questions_per_doc) + 1)
        
        if show_progress:
            console.print(f"\n[bold cyan]🤖 Generating Synthetic Q&A Pairs[/bold cyan]")
            console.print(f"  Target: {target_count} questions from {docs_needed} documents\n")
        
        # Process documents with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not show_progress
        ) as progress:
            
            task = progress.add_task(
                f"Generating questions...",
                total=docs_needed
            )
            
            for i, doc in enumerate(documents[:docs_needed]):
                # Stop if we've reached target count
                if len(qa_pairs) >= target_count:
                    break
                
                # Generate Q&A pairs from this document
                try:
                    pairs = self._generate_from_document(doc)
                    
                    # Validate and add to results
                    for pair in pairs:
                        if self._validate_qa_pair(pair):
                            qa_pairs.append(pair)
                        else:
                            self.stats["invalid_qa_pairs"] += 1
                    
                except Exception as e:
                    self.stats["failed_generations"] += 1
                    if show_progress:
                        console.print(f"  [yellow]⚠️  Failed to generate from doc {doc['id']}: {e}[/yellow]")
                
                progress.update(task, advance=1)
        
        # Update final stats
        self.stats["total_questions"] = len(qa_pairs)
        
        if show_progress:
            console.print(f"\n[green]✓[/green] Generated {len(qa_pairs)} Q&A pairs")
            if self.stats["failed_generations"] > 0:
                console.print(f"  [yellow]⚠️  Failed: {self.stats['failed_generations']} documents[/yellow]")
            if self.stats["invalid_qa_pairs"] > 0:
                console.print(f"  [yellow]⚠️  Invalid: {self.stats['invalid_qa_pairs']} Q&A pairs[/yellow]")
        
        return qa_pairs[:target_count]  # Return exactly target_count
    
    def _generate_from_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs from a single document using Groq.
        
        Args:
            document: Document dict with 'id', 'text', 'metadata'
            
        Returns:
            List of Q&A pairs for this document
        """
        # Truncate document if too long (to avoid token limits)
        max_doc_length = 2000  # chars
        doc_text = document["text"][:max_doc_length]
        
        # Create prompt for Q&A generation
        prompt = f"""You are a question generation assistant. Based on the document below, generate {self.questions_per_doc} diverse, high-quality questions that test understanding of the content.

For each question, also provide the correct answer based on the document.

Document:
{doc_text}

Generate {self.questions_per_doc} questions with these types:
1. Factual question (tests specific details)
2. Conceptual question (tests understanding of main ideas)

Return your response in this exact JSON format:
{{
    "qa_pairs": [
        {{"question": "Your question here?", "answer": "Your answer here"}},
        {{"question": "Your question here?", "answer": "Your answer here"}}
    ]
}}

Generate ONLY valid JSON, no other text."""

        # Call Groq API (model rotation handled by wrapper)
        response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a question generation expert. Generate diverse, clear questions that test document comprehension. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=1000
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON (sometimes LLM adds extra text)
        try:
            # Find JSON in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            
            data = json.loads(json_str)
            qa_pairs = data.get("qa_pairs", [])
            
            # Add document metadata to each pair
            for pair in qa_pairs:
                pair["document_id"] = document["id"]
                pair["metadata"] = {
                    "generated_at": datetime.now().isoformat(),
                    "source_text_preview": doc_text[:100] + "..."
                }
            
            return qa_pairs
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text[:200]}")
    
    def _validate_qa_pair(self, qa_pair: Dict[str, Any]) -> bool:
        """
        Validate a Q&A pair for quality.
        
        Args:
            qa_pair: Dict with 'question' and 'answer' keys
            
        Returns:
            True if valid, False otherwise
        """
        question = qa_pair.get("question", "").strip()
        answer = qa_pair.get("answer", "").strip()
        
        # Check non-empty
        if not question or not answer:
            return False
        
        # Check question length (reasonable bounds)
        if len(question) < 10 or len(question) > 500:
            return False
        
        # Check answer length
        if len(answer) < 5 or len(answer) > 1000:
            return False
        
        # Check question ends with question mark
        if not question.endswith("?"):
            return False
        
        return True
    
    def save_to_file(
        self,
        qa_pairs: List[Dict[str, Any]],
        output_path: str | Path = "/reports/synthetic_qa.json"
    ):
        """
        Save Q&A pairs to JSON file.
        
        Args:
            qa_pairs: List of Q&A pairs to save
            output_path: Path to output file (default: synthetic_qa.json)
        """
        output_path = Path(output_path)
        
        # Create reports folder if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create output data with metadata
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_pairs": len(qa_pairs),
                "statistics": self.stats
            },
            "qa_pairs": qa_pairs
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓[/green] Saved {len(qa_pairs)} Q&A pairs to: {output_path}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get generation statistics."""
        return self.stats.copy()

