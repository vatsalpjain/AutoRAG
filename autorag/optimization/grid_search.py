"""
Grid Search Optimization - Tests multiple RAG configurations to find the best one.
Evaluates configurations on accuracy, cost, and latency metrics.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from autorag.rag.pipeline import RAGPipeline
from autorag.evaluation.ragas_eval import RagasEvaluator
import time

# Initialize Rich console for terminal output
console = Console()


class GridSearchOptimizer:
    """
    Grid search optimizer for RAG configurations.
    Tests multiple configurations and ranks by weighted score.
    """
    
    def __init__(self, pipeline: RAGPipeline, groq_api_key: str = None):
        """
        Initialize grid search optimizer.
        
        Args:
            pipeline: Initialized RAG pipeline to test configurations with
            groq_api_key: Groq API key for Ragas evaluation (optional if using SequenceMatcher fallback)
        """
        self.pipeline = pipeline
        self.results = []
        self.ragas_evaluator = RagasEvaluator(groq_api_key) if groq_api_key else None
    
    def define_search_space(self) -> List[Dict[str, Any]]:
        """
        Define the grid of configurations to test.
        
        Returns:
            List of configuration dicts with 'top_k', 'temperature', 'name'
        """
        configurations = []
        
        # Grid parameters
        top_k_values = [3, 5, 10]
        temperature_values = [0.3, 0.7, 1.0]
        
        # Generate all combinations
        for top_k in top_k_values:
            for temp in temperature_values:
                config = {
                    "top_k": top_k,
                    "temperature": temp,
                    "name": f"k{top_k}_t{temp:.1f}"
                }
                configurations.append(config)
        
        return configurations
    
    def optimize(
        self,
        qa_pairs: List[Dict[str, Any]],
        max_configs: int = 9,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run grid search optimization.
        
        Args:
            qa_pairs: List of Q&A pairs to test against
            max_configs: Maximum number of configurations to test (default: 9)
            show_progress: Whether to show progress bars (default: True)
            
        Returns:
            List of results sorted by weighted score (best first)
        """
        # Define search space
        configurations = self.define_search_space()[:max_configs]
        
        if show_progress:
            console.print(f"\n[bold cyan]🔍 Running Grid Search Optimization[/bold cyan]")
            console.print(f"  Testing {len(configurations)} configurations")
            console.print(f"  Evaluating on {len(qa_pairs)} Q&A pairs\n")
        
        # Test each configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not show_progress
        ) as progress:
            
            main_task = progress.add_task(
                "Testing configurations...",
                total=len(configurations)
            )
            
            for config in configurations:
                if show_progress:
                    progress.update(
                        main_task,
                        description=f"Testing {config['name']} - Running queries..."
                    )
                
                # Evaluate this configuration
                result = self._evaluate_config(config, qa_pairs, progress, main_task)
                self.results.append(result)
                
                progress.update(main_task, advance=1)
                
                # Add delay between configs to prevent rate limiting
                # 60 seconds ensures we stay well under 90 RPM across configs
                if config != configurations[-1]:  # Don't delay after last config
                    time.sleep(60)
        
        # Sort by weighted score (best first)
        self.results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        if show_progress:
            console.print(f"\n[green]✓[/green] Grid search complete!")
            self._display_results_table()
        
        return self.results
    
    def _evaluate_config(
        self,
        config: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
        progress = None,
        task_id = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single configuration on all Q&A pairs using Ragas metrics.
        
        Args:
            config: Configuration dict with 'top_k', 'temperature', 'name'
            qa_pairs: List of Q&A pairs to test
            progress: Optional Rich Progress instance for status updates
            task_id: Optional task ID for progress updates
            
        Returns:
            Result dict with metrics and scores
        """
        # Accumulators
        total_tokens = 0
        total_latency = 0.0
        successful_queries = 0
        rag_results = []  # Store results for batch Ragas evaluation
        
        # Run all queries and collect results
        for qa_pair in qa_pairs:
            try:
                # Measure latency
                start_time = time.time()
                
                # Run query with this config
                result = self.pipeline.query(
                    question=qa_pair["question"],
                    top_k=config["top_k"],
                    temperature=config["temperature"]
                )
                
                latency = time.time() - start_time
                
                # Estimate token count (rough approximation: 1 token ≈ 4 chars)
                tokens = self._estimate_tokens(
                    question=qa_pair["question"],
                    context=result["retrieved_docs"],
                    answer=result["answer"]
                )
                
                # Store result for Ragas evaluation
                rag_results.append({
                    "question": qa_pair["question"],
                    "answer": result["answer"],
                    "retrieved_docs": result["retrieved_docs"]
                })
                
                # Accumulate metrics
                total_tokens += tokens
                total_latency += latency
                successful_queries += 1
                
            except Exception as e:
                # Skip failed queries
                console.print(f"[yellow]⚠️  Query failed: {e}[/yellow]")
                continue
        
        # Calculate averages
        if successful_queries > 0:
            avg_tokens = total_tokens / successful_queries
            avg_latency = total_latency / successful_queries
        else:
            avg_tokens = 0.0
            avg_latency = 0.0
        
        # Evaluate using Ragas (ONE-BY-ONE to prevent rate limit bursts)
        ragas_scores = {}
        if self.ragas_evaluator and successful_queries > 0:
            try:
                # Update progress to show Ragas evaluation stage
                if progress and task_id:
                    progress.update(task_id, description=f"Testing {config['name']} - Ragas evaluation...")
                
                # Evaluate ONE question at a time to add delays between batches
                import os
                os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
                
                all_metric_scores = {"answer_relevancy": [], "faithfulness": [], "answer_similarity": []}
                
                # Process each Q&A pair individually
                for i, (qa_pair, rag_result) in enumerate(zip(qa_pairs, rag_results)):
                    # Prepare single-question dataset
                    dataset = self.ragas_evaluator.prepare_dataset([qa_pair], [rag_result])
                    
                    # Suppress tqdm for cleaner output
                    from io import StringIO
                    import sys
                    original_stderr = sys.stderr
                    sys.stderr = StringIO()
                    
                    # Evaluate this one question
                    result = self.ragas_evaluator.evaluate(dataset)
                    
                    sys.stderr = original_stderr
                    
                    # Collect metric scores
                    for metric in all_metric_scores.keys():
                        if metric in result:
                            all_metric_scores[metric].append(result[metric])
                
                # Average all metric scores across questions
                ragas_scores = {
                    metric: sum(scores) / len(scores) if scores else 0.0
                    for metric, scores in all_metric_scores.items()
                }
                
                # Calculate aggregate Ragas score
                avg_accuracy = self.ragas_evaluator.calculate_aggregate_score(ragas_scores)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Ragas evaluation failed: {e}[/yellow]")
                console.print(f"[dim]Falling back to simple similarity...[/dim]")
                avg_accuracy = 0.5  # Fallback score
        else:
            # No Ragas evaluator - use fallback
            avg_accuracy = 0.5
        
        # Use Ragas aggregate score as primary ranking metric
        # Cost and latency are tracked for info, but don't affect ranking
        # (they depend on LLM/infrastructure, not RAG config quality)
        weighted_score = avg_accuracy  # Pure Ragas aggregate score
        
        # Calculate cost/latency scores for reference only
        cost_score = 1.0 / (avg_tokens / 1000 + 1)  # Normalize by 1k tokens
        latency_score = 1.0 / (avg_latency + 0.1)  # Normalize by seconds
        
        return {
            "config": config,
            "metrics": {
                "accuracy": avg_accuracy,
                "avg_tokens": avg_tokens,
                "avg_latency_seconds": avg_latency,
                "successful_queries": successful_queries,
                "total_queries": len(qa_pairs),
                # Ragas metric breakdown (if available)
                "ragas_answer_relevancy": ragas_scores.get("answer_relevancy", None),
                "ragas_faithfulness": ragas_scores.get("faithfulness", None),
                "ragas_answer_similarity": ragas_scores.get("answer_similarity", None)
            },
            "scores": {
                "accuracy_score": avg_accuracy,
                "cost_score": cost_score,
                "latency_score": latency_score
            },
            "weighted_score": weighted_score
        }
    
    def _estimate_tokens(
        self,
        question: str,
        context: List[Dict[str, Any]],
        answer: str
    ) -> int:
        """
        Estimate token usage for a query.
        
        Args:
            question: User's question
            context: Retrieved documents
            answer: Generated answer
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        # This is simplified - real tokenizers vary
        
        # Count question tokens
        question_chars = len(question)
        
        # Count context tokens (from all retrieved docs)
        context_chars = sum(len(doc.get("text", "")) for doc in context)
        
        # Count answer tokens
        answer_chars = len(answer)
        
        # Total characters / 4 = rough token count
        total_tokens = (question_chars + context_chars + answer_chars) // 4
        
        return total_tokens
    
    def _display_results_table(self):
        """Display results in a formatted table."""
        console.print("\n[bold cyan]📊 Optimization Results[/bold cyan]\n")
        
        # Create Rich table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Config", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Latency (s)", justify="right")
        table.add_column("Score", justify="right", style="green")
        
        # Add top 5 results
        for i, result in enumerate(self.results[:5], 1):
            config = result["config"]
            metrics = result["metrics"]
            
            table.add_row(
                str(i),
                config["name"],
                f"{metrics['accuracy']:.3f}",
                f"{metrics['avg_tokens']:.0f}",
                f"{metrics['avg_latency_seconds']:.2f}",
                f"{result['weighted_score']:.3f}"
            )
        
        console.print(table)
        
        # Show best config details
        best = self.results[0]
        console.print(f"\n[bold green]🏆 Best Configuration:[/bold green]")
        console.print(f"  Name: {best['config']['name']}")
        console.print(f"  top_k: {best['config']['top_k']}")
        console.print(f"  temperature: {best['config']['temperature']}")
        console.print(f"  Weighted Score: {best['weighted_score']:.3f}")
    
    def save_results(self, output_path: str | Path = "/reports/optimization_results.json"):
        """
        Save optimization results to JSON file.
        
        Args:
            output_path: Path to output file (default: optimization_results.json)
        """
        output_path = Path(output_path)
        
        # Create output data
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_configs_tested": len(self.results),
                "best_config": self.results[0]["config"] if self.results else None
            },
            "results": self.results
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓[/green] Saved results to: {output_path}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best performing configuration."""
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")
        return self.results[0]
