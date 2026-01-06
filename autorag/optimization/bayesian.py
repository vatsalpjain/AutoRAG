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
from autorag.evaluation.evaluator_factory import get_evaluator

# Initialize Rich console for terminal output
console = Console()

# Suppress Optuna's default logging (we use Rich for output)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class EarlyStoppingCallback:
    """
    Optuna callback that stops optimization when score stops improving.
    Saves LLM calls by stopping early when convergence is detected.
    """
    
    def __init__(self, patience: int = 3, min_trials: int = 5):
        """
        Initialize early stopping.
        
        Args:
            patience: Stop if no improvement for this many consecutive trials
            min_trials: Minimum trials before early stopping kicks in
        """
        self.patience = patience
        self.min_trials = min_trials
        self.best_value = float('-inf')
        self.trials_without_improvement = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Called after each trial. Raises StopIteration to stop optimization."""
        current_best = study.best_value
        
        # Skip early stopping for first few trials (exploration phase)
        if len(study.trials) < self.min_trials:
            return
        
        # Check if improved
        if current_best > self.best_value:
            self.best_value = current_best
            self.trials_without_improvement = 0
            console.print(f"  [green]↑ New best: {current_best:.3f}[/green]")
        else:
            self.trials_without_improvement += 1
            console.print(f"  [dim]No improvement ({self.trials_without_improvement}/{self.patience})[/dim]")
        
        # Stop if patience exceeded
        if self.trials_without_improvement >= self.patience:
            console.print(f"\n[yellow]⚡ Early stopping! No improvement for {self.patience} trials.[/yellow]")
            study.stop()


class BayesianOptimizer:
    """
    Bayesian optimizer for RAG configurations using Optuna.
    Intelligently samples configurations to find the best one faster than grid search.
    """
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        llm_provider: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
        evaluation_method: str = "custom"
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            pipeline: Initialized RAG pipeline to test configurations with
            llm_provider: LLM provider (groq, openai, openrouter) for evaluation
            llm_api_key: API key for the LLM provider
            llm_model: Optional model name
            evaluation_method: Evaluation method - 'custom' or 'ragas'
        """
        self.pipeline = pipeline
        self.results = []
        self.study = None  # Optuna study object
        self.evaluation_method = evaluation_method
        self.evaluator = get_evaluator(
            method=evaluation_method,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        ) if llm_api_key else None
        # Track actual method used (may differ if fallback)
        if self.evaluator:
            self.evaluation_method = self.evaluator.get_evaluation_method()
        
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
            console.print(f"  Search space: top_k ∈ [3,7,10], temperature ∈ [0.3,0.7,1.0]\n")
        
        # Create Optuna study (maximize score)
        self.study = optuna.create_study(
            direction="maximize",
            study_name="autorag_optimization",
            sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
        )
        
        # Early stopping callback - stops if no improvement for 3 trials
        early_stop_callback = EarlyStoppingCallback(patience=3, min_trials=5)
        
        # Run optimization with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not show_progress,
            refresh_per_second=2  # Keep spinner animated during LLM calls
        ) as progress:
            
            self._progress = progress
            self._task_id = progress.add_task(
                "Optimizing configurations...",
                total=n_trials
            )
            
            # Run Optuna optimization with early stopping
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                callbacks=[early_stop_callback],
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
        
        # Optuna suggests parameters from SAME search space as Grid Search
        # Uses categorical suggestions for fair comparison
        top_k = trial.suggest_categorical("top_k", [3, 5, 10])
        temperature = trial.suggest_categorical("temperature", [0.3, 0.7, 1.0])
        
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
        
        # Evaluate using AutoRAGEvaluator
        eval_scores = {}
        avg_accuracy = 0.5  # Default fallback
        
        if self.evaluator and successful_queries > 0:
            try:
                all_scores = {"answer_relevancy": [], "faithfulness": [], "answer_similarity": [], "context_recall": []}
                
                for qa_pair, rag_result in zip(self._qa_pairs, rag_results):
                    # Build context string
                    context = " ".join([doc["text"] for doc in rag_result.get("retrieved_docs", [])])
                    
                    # Evaluate single pair
                    scores = self.evaluator.evaluate(
                        question=qa_pair["question"],
                        answer=rag_result["answer"],
                        context=context,
                        reference=qa_pair["answer"]
                    )
                    
                    # Collect scores
                    for metric, value in scores.items():
                        if metric in all_scores and value is not None:
                            all_scores[metric].append(value)
                
                # Average scores
                eval_scores = {
                    metric: sum(values) / len(values) if values else 0.0
                    for metric, values in all_scores.items()
                }
                
                avg_accuracy = self.evaluator.calculate_aggregate_score(eval_scores)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Evaluation failed: {e}[/yellow]")
                avg_accuracy = 0.5
        
        return {
            "config": config,
            "metrics": {
                "accuracy": avg_accuracy,
                "avg_tokens": avg_tokens,
                "avg_latency_seconds": avg_latency,
                "successful_queries": successful_queries,
                "total_queries": len(self._qa_pairs),
                "answer_relevancy": eval_scores.get("answer_relevancy"),
                "faithfulness": eval_scores.get("faithfulness"),
                "answer_similarity": eval_scores.get("answer_similarity"),
                "context_recall": eval_scores.get("context_recall")
            },
            "scores": {
                "accuracy_score": avg_accuracy,
                "cost_score": 1.0 / (avg_tokens / 1000 + 1),
                "latency_score": 1.0 / (avg_latency + 0.1)
            },
            "weighted_score": avg_accuracy
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
                "evaluation_method": self.evaluation_method,
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
