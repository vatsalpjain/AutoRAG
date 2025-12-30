"""
Bayesian Optimization using Optuna - Intelligently searches for optimal RAG configurations.
Uses probabilistic models to find the best config in fewer trials than grid search.
"""
import json
import time
import optuna
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from autorag.rag.pipeline import RAGPipeline
from autorag.evaluation.ragas_eval import RagasEvaluator

# Initialize Rich console for terminal output
console = Console()

# Suppress Optuna's default logging (we use Rich for output)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BayesianOptimizer:
    """
    Bayesian optimizer for RAG configurations using Optuna.
    Intelligently samples configurations to find the best one faster than grid search.
    """
    
    def __init__(self, pipeline: RAGPipeline, groq_api_key: str = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            pipeline: Initialized RAG pipeline to test configurations with
            groq_api_key: Groq API key for Ragas evaluation
        """
        self.pipeline = pipeline
        self.results = []
        self.study = None  # Optuna study object
        self.ragas_evaluator = RagasEvaluator(groq_api_key) if groq_api_key else None
        
        # Will be set during optimize()
        self._qa_pairs = None
        self._progress = None
        self._task_id = None
        self._trial_count = 0
        self._total_trials = 0
    
    def optimize(
        self,
        qa_pairs: list,
        n_trials: int = 20,
        show_progress: bool = True
    ) -> list:
        """
        Run Bayesian optimization using Optuna.
        
        Args:
            qa_pairs: List of Q&A pairs to test against
            n_trials: Number of trials to run (default: 20)
            show_progress: Whether to show progress bars (default: True)
            
        Returns:
            List of results sorted by weighted score (best first)
        """
        # Store for use in objective function
        self._qa_pairs = qa_pairs
        self._trial_count = 0
        self._total_trials = n_trials
        
        if show_progress:
            console.print(f"\n[bold cyan]🧠 Running Bayesian Optimization (Optuna)[/bold cyan]")
            console.print(f"  Maximum trials: {n_trials}")
            console.print(f"  Evaluating on {len(qa_pairs)} Q&A pairs")
            console.print(f"  Search space: top_k ∈ [3, 10], temperature ∈ [0.1, 1.0]\n")
        
        # Create Optuna study (maximize RAGAS score)
        self.study = optuna.create_study(
            direction="maximize",
            study_name="autorag_optimization",
            sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
        )
        
        # Run optimization with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not show_progress
        ) as progress:
            
            self._progress = progress
            self._task_id = progress.add_task(
                "Optimizing configurations...",
                total=n_trials
            )
            
            # Run Optuna optimization
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                show_progress_bar=False  # We use Rich instead
            )
        
        # Sort results by weighted score (best first)
        self.results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        if show_progress:
            console.print(f"\n[green]✓[/green] Bayesian optimization complete!")
            console.print(f"  Best score: [green]{self.study.best_value:.3f}[/green]")
            console.print(f"  Best params: top_k={self.study.best_params['top_k']}, "
                         f"temperature={self.study.best_params['temperature']:.2f}")
            self._display_results_table()
        
        return self.results
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function. Called for each trial.
        
        Args:
            trial: Optuna trial object for suggesting parameters
            
        Returns:
            Score to maximize (RAGAS aggregate score)
        """
        self._trial_count += 1
        
        # Optuna suggests parameters to try
        top_k = trial.suggest_int("top_k", 3, 10)
        temperature = trial.suggest_float("temperature", 0.1, 1.0)
        
        # Create config dict
        config = {
            "top_k": top_k,
            "temperature": round(temperature, 2),
            "name": f"trial_{self._trial_count}_k{top_k}_t{temperature:.2f}"
        }
        
        # Update progress
        if self._progress and self._task_id:
            self._progress.update(
                self._task_id,
                description=f"Trial {self._trial_count}/{self._total_trials}: k={top_k}, t={temperature:.2f}",
                completed=self._trial_count
            )
        
        # Evaluate this configuration
        result = self._evaluate_config(config)
        self.results.append(result)
        
        # Add delay between configs to prevent rate limiting
        # 60 seconds ensures we stay well under 90 RPM across configs
        if self._trial_count < self._total_trials:
            time.sleep(60)
        
        # Return score for Optuna to maximize
        return result["weighted_score"]
    
    def _evaluate_config(self, config: dict) -> dict:
        """
        Evaluate a single configuration on all Q&A pairs using Ragas metrics.
        
        Args:
            config: Configuration dict with 'top_k', 'temperature', 'name'
            
        Returns:
            Result dict with metrics and scores
        """
        # Accumulators
        total_tokens = 0
        total_latency = 0.0
        successful_queries = 0
        rag_results = []
        
        # Run all queries and collect results
        for qa_pair in self._qa_pairs:
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
                
                # Estimate token count
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
                console.print(f"[yellow]⚠️  Query failed: {e}[/yellow]")
                continue
        
        # Calculate averages
        avg_tokens = total_tokens / successful_queries if successful_queries > 0 else 0.0
        avg_latency = total_latency / successful_queries if successful_queries > 0 else 0.0
        
        # Evaluate using Ragas
        ragas_scores = {}
        avg_accuracy = 0.5  # Default fallback
        
        if self.ragas_evaluator and successful_queries > 0:
            try:
                import os
                os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
                
                all_metric_scores = {"answer_relevancy": [], "faithfulness": [], "answer_similarity": []}
                
                # Process each Q&A pair individually
                for i, (qa_pair, rag_result) in enumerate(zip(self._qa_pairs, rag_results)):
                    dataset = self.ragas_evaluator.prepare_dataset([qa_pair], [rag_result])
                    
                    # Evaluate (stderr NOT suppressed - see actual errors)
                    result = self.ragas_evaluator.evaluate(dataset)
                    
                    # Collect metric scores
                    for metric in all_metric_scores.keys():
                        if metric in result:
                            all_metric_scores[metric].append(result[metric])
                
                # Average all metric scores
                ragas_scores = {
                    metric: sum(scores) / len(scores) if scores else 0.0
                    for metric, scores in all_metric_scores.items()
                }
                
                avg_accuracy = self.ragas_evaluator.calculate_aggregate_score(ragas_scores)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Ragas evaluation failed: {e}[/yellow]")
                avg_accuracy = 0.5
        
        return {
            "config": config,
            "metrics": {
                "accuracy": avg_accuracy,
                "avg_tokens": avg_tokens,
                "avg_latency_seconds": avg_latency,
                "successful_queries": successful_queries,
                "total_queries": len(self._qa_pairs),
                "ragas_answer_relevancy": ragas_scores.get("answer_relevancy"),
                "ragas_faithfulness": ragas_scores.get("faithfulness"),
                "ragas_answer_similarity": ragas_scores.get("answer_similarity")
            },
            "scores": {
                "accuracy_score": avg_accuracy,
                "cost_score": 1.0 / (avg_tokens / 1000 + 1),
                "latency_score": 1.0 / (avg_latency + 0.1)
            },
            "weighted_score": avg_accuracy  # Pure Ragas aggregate score
        }
    
    def _estimate_tokens(self, question: str, context: list, answer: str) -> int:
        """Estimate token usage for a query (rough approximation: 1 token ≈ 4 chars)."""
        question_chars = len(question)
        context_chars = sum(len(doc.get("text", "")) for doc in context)
        answer_chars = len(answer)
        return (question_chars + context_chars + answer_chars) // 4
    
    def _display_results_table(self):
        """Display results in a formatted table."""
        from rich.table import Table
        
        console.print("\n[bold cyan]📊 Optimization Results (Top 5)[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Config", style="cyan")
        table.add_column("top_k", justify="right")
        table.add_column("temp", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Score", justify="right", style="green")
        
        for i, result in enumerate(self.results[:5], 1):
            config = result["config"]
            metrics = result["metrics"]
            
            table.add_row(
                str(i),
                config["name"],
                str(config["top_k"]),
                f"{config['temperature']:.2f}",
                f"{metrics['accuracy']:.3f}",
                f"{result['weighted_score']:.3f}"
            )
        
        console.print(table)
        
        # Show best config
        best = self.results[0]
        console.print(f"\n[bold green]🏆 Best Configuration:[/bold green]")
        console.print(f"  top_k: {best['config']['top_k']}")
        console.print(f"  temperature: {best['config']['temperature']}")
        console.print(f"  Score: {best['weighted_score']:.3f}")
    
    def save_results(self, output_path: str | Path = "reports/optimization_results.json"):
        """
        Save optimization results to JSON file.
        
        Args:
            output_path: Path to output file (default: optimization_results.json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "optimization_method": "bayesian",
                "total_trials": len(self.results),
                "best_config": self.results[0]["config"] if self.results else None,
                "best_score": self.study.best_value if self.study else None
            },
            "results": self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓[/green] Saved results to: {output_path}")
    
    def get_best_config(self) -> dict:
        """Get the best performing configuration."""
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")
        return self.results[0]
