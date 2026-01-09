"""
Bayesian Optimization using Optuna - Intelligently searches for optimal RAG configurations.
Uses probabilistic models to find the best config with smart indexing caching.
"""
import json
import optuna
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from autorag.rag.pipeline import RAGPipeline
from autorag.evaluation.evaluator_factory import get_evaluator

# Initialize Rich console for terminal output
console = Console()

# Suppress Optuna's default logging (we use Rich instead)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class EarlyStoppingCallback:
    """
    Optuna callback that stops optimization when score stops improving.
    Saves LLM calls by stopping early when convergence is detected.
    """
    
    def __init__(self, patience: int = 3, min_trials: int = 5):
        self.patience = patience
        self.min_trials = min_trials
        self.best_value = float('-inf')
        self.trials_without_improvement = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        current_best = study.best_value
        
        if len(study.trials) < self.min_trials:
            return
        
        if current_best > self.best_value:
            self.best_value = current_best
            self.trials_without_improvement = 0
            console.print(f"  [green]↑ New best: {current_best:.3f}[/green]")
        else:
            self.trials_without_improvement += 1
            console.print(f"  [dim]No improvement ({self.trials_without_improvement}/{self.patience})[/dim]")
        
        if self.trials_without_improvement >= self.patience:
            console.print(f"\n[yellow]⚡ Early stopping! No improvement for {self.patience} trials.[/yellow]")
            study.stop()


class BayesianOptimizer:
    """
    Bayesian optimizer for RAG configurations using Optuna.
    Uses two-phase optimization with intelligent caching of indexed pipelines.
    """
    
    def __init__(
        self,
        llm_provider: str,
        llm_api_key: str,
        llm_model: str = None,
        evaluation_method: str = "custom",
        rag_config: Dict[str, Any] = None,
        documents: List[Dict[str, Any]] = None
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            llm_provider: LLM provider (groq, openai, openrouter)
            llm_api_key: API key for the LLM provider
            llm_model: Optional model name
            evaluation_method: Evaluation method - 'custom' or 'ragas'
            rag_config: RAG parameter search space from config.yaml
            documents: Documents to index
        """
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.documents = documents or []
        self.results = []
        self.study = None
        self.evaluation_method = evaluation_method
        
        # Parse RAG config with defaults
        rag_config = rag_config or {}
        self.chunk_sizes = rag_config.get("chunk_size", [500])
        self.chunk_overlaps = rag_config.get("chunk_overlap", [50])
        self.embedding_models = rag_config.get("embedding_model", ["all-MiniLM-L6-v2"])
        self.top_k_values = rag_config.get("top_k", [3, 5, 10])
        self.temperature_values = rag_config.get("temperature", [0.3, 0.7, 1.0])
        
        # Initialize evaluator
        self.evaluator = get_evaluator(
            method=evaluation_method,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        ) if llm_api_key else None
        
        if self.evaluator:
            self.evaluation_method = self.evaluator.get_evaluation_method()
        
        # Pipeline cache: (chunk_size, overlap, model) -> pipeline
        self._pipeline_cache: Dict[tuple, RAGPipeline] = {}
        self._current_indexing_key = None
        
        # Progress tracking
        self._qa_pairs = None
        self._progress = None
        self._task_id = None
        self._trial_count = 0
        self._total_trials = 0
    
    def _get_or_create_pipeline(
        self, 
        chunk_size: int, 
        chunk_overlap: int, 
        embedding_model: str
    ) -> RAGPipeline:
        """
        Get cached pipeline or create and index a new one.
        This avoids redundant re-indexing when Optuna suggests same indexing params.
        """
        key = (chunk_size, chunk_overlap, embedding_model)
        
        if key not in self._pipeline_cache:
            # Create new pipeline
            model_short = embedding_model.replace("/", "_").replace("-", "_")
            collection_name = f"autorag_c{chunk_size}_o{chunk_overlap}_{model_short}"
            
            console.print(f"\n  [bold yellow]📦 New indexing config detected[/bold yellow]")
            console.print(f"    chunk_size={chunk_size}, overlap={chunk_overlap}, model={embedding_model}")
            
            pipeline = RAGPipeline(
                llm_provider=self.llm_provider,
                llm_api_key=self.llm_api_key,
                llm_model=self.llm_model,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name=collection_name
            )
            
            # Clear and index
            console.print("    Clearing index...", end=" ")
            pipeline.clear_index()
            console.print("[green]✓[/green]")
            
            console.print(f"    Indexing {len(self.documents)} documents...", end=" ")
            pipeline.index_documents(self.documents)
            console.print("[green]✓[/green]")
            
            self._pipeline_cache[key] = pipeline
        
        return self._pipeline_cache[key]
    
    def optimize(
        self,
        qa_pairs: List[Dict[str, Any]],
        n_trials: int = 20,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run Bayesian optimization using Optuna.
        
        Args:
            qa_pairs: List of Q&A pairs to test against
            n_trials: Number of trials to run
            show_progress: Whether to show progress bars
            
        Returns:
            List of results sorted by weighted score (best first)
        """
        self._qa_pairs = qa_pairs
        self._trial_count = 0
        self._total_trials = n_trials
        
        if show_progress:
            console.print(f"\n[bold cyan]🧠 Bayesian Optimization (Optuna)[/bold cyan]")
            console.print(f"  Maximum trials: {n_trials}")
            console.print(f"  Evaluating on {len(qa_pairs)} Q&A pairs")
            console.print(f"  Search space:")
            console.print(f"    chunk_size: {self.chunk_sizes}")
            console.print(f"    chunk_overlap: {self.chunk_overlaps}")
            console.print(f"    embedding_model: {self.embedding_models}")
            console.print(f"    top_k: {self.top_k_values}")
            console.print(f"    temperature: {self.temperature_values}\n")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name="autorag_optimization",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Early stopping
        early_stop_callback = EarlyStoppingCallback(patience=3, min_trials=5)
        
        # Run optimization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=not show_progress,
            refresh_per_second=2
        ) as progress:
            
            self._progress = progress
            self._task_id = progress.add_task(
                "Optimizing configurations...",
                total=n_trials
            )
            
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                callbacks=[early_stop_callback],
                show_progress_bar=False
            )
        
        # Sort results
        self.results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        if show_progress:
            console.print(f"\n[green]✓[/green] Bayesian optimization complete!")
            console.print(f"  Pipelines cached: {len(self._pipeline_cache)} (re-indexing avoided)")
            console.print(f"  Best score: [green]{self.study.best_value:.3f}[/green]")
            self._display_results_table()
        
        return self.results
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function. Called for each trial."""
        self._trial_count += 1
        
        # Suggest ALL parameters (including indexing params)
        chunk_size = trial.suggest_categorical("chunk_size", self.chunk_sizes)
        chunk_overlap = trial.suggest_categorical("chunk_overlap", self.chunk_overlaps)
        embedding_model = trial.suggest_categorical("embedding_model", self.embedding_models)
        top_k = trial.suggest_categorical("top_k", self.top_k_values)
        temperature = trial.suggest_categorical("temperature", self.temperature_values)
        
        # Get or create pipeline (with caching)
        pipeline = self._get_or_create_pipeline(chunk_size, chunk_overlap, embedding_model)
        
        # Build config
        config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "top_k": top_k,
            "temperature": round(temperature, 2),
            "name": f"trial_{self._trial_count}_c{chunk_size}_k{top_k}_t{temperature:.1f}"
        }
        
        # Update progress
        if self._progress and self._task_id:
            self._progress.update(
                self._task_id,
                description=f"Trial {self._trial_count}/{self._total_trials}: k={top_k}, t={temperature:.1f}",
                completed=self._trial_count
            )
        
        # Evaluate this configuration
        result = self._evaluate_config(config, pipeline)
        self.results.append(result)
        
        return result["weighted_score"]
    
    def _evaluate_config(
        self, 
        config: Dict[str, Any], 
        pipeline: RAGPipeline
    ) -> Dict[str, Any]:
        """Evaluate a single configuration on all Q&A pairs."""
        successful_queries = 0
        rag_results = []
        
        for qa_pair in self._qa_pairs:
            try:
                result = pipeline.query(
                    question=qa_pair["question"],
                    top_k=config["top_k"],
                    temperature=config["temperature"]
                )
                
                rag_results.append({
                    "question": qa_pair["question"],
                    "answer": result["answer"],
                    "retrieved_docs": result["retrieved_docs"]
                })
                successful_queries += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Query failed: {e}[/yellow]")
                continue
        
        # Evaluate
        eval_scores = {}
        avg_accuracy = 0.5
        
        if self.evaluator and successful_queries > 0:
            try:
                all_scores = {
                    "answer_relevancy": [], 
                    "faithfulness": [], 
                    "answer_similarity": [], 
                    "context_recall": []
                }
                
                for qa_pair, rag_result in zip(self._qa_pairs, rag_results):
                    context = " ".join([doc["text"] for doc in rag_result.get("retrieved_docs", [])])
                    
                    scores = self.evaluator.evaluate(
                        question=qa_pair["question"],
                        answer=rag_result["answer"],
                        context=context,
                        reference=qa_pair["answer"]
                    )
                    
                    for metric, value in scores.items():
                        if metric in all_scores and value is not None:
                            all_scores[metric].append(value)
                
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
                "successful_queries": successful_queries,
                "total_queries": len(self._qa_pairs),
                "answer_relevancy": eval_scores.get("answer_relevancy"),
                "faithfulness": eval_scores.get("faithfulness"),
                "answer_similarity": eval_scores.get("answer_similarity"),
                "context_recall": eval_scores.get("context_recall")
            },
            "weighted_score": avg_accuracy
        }
    
    def _display_results_table(self):
        """Display results in a formatted table."""
        console.print("\n[bold cyan]📊 Optimization Results (Top 5)[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Config", style="cyan")
        table.add_column("Chunk", justify="right")
        table.add_column("top_k", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Score", justify="right", style="green")
        
        for i, result in enumerate(self.results[:5], 1):
            config = result["config"]
            metrics = result["metrics"]
            
            table.add_row(
                str(i),
                config["name"],
                str(config["chunk_size"]),
                str(config["top_k"]),
                f"{metrics['accuracy']:.3f}",
                f"{result['weighted_score']:.3f}"
            )
        
        console.print(table)
        
        # Show best config
        best = self.results[0]
        console.print(f"\n[bold green]🏆 Best Configuration:[/bold green]")
        console.print(f"  chunk_size: {best['config']['chunk_size']}")
        console.print(f"  chunk_overlap: {best['config']['chunk_overlap']}")
        console.print(f"  embedding_model: {best['config']['embedding_model']}")
        console.print(f"  top_k: {best['config']['top_k']}")
        console.print(f"  temperature: {best['config']['temperature']}")
        console.print(f"  Score: {best['weighted_score']:.3f}")
    
    def save_results(self, output_path: str | Path = "reports/optimization_results.json"):
        """Save optimization results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "optimization_method": "bayesian",
                "evaluation_method": self.evaluation_method,
                "total_trials": len(self.results),
                "pipelines_cached": len(self._pipeline_cache),
                "best_config": self.results[0]["config"] if self.results else None,
                "best_score": self.study.best_value if self.study else None
            },
            "results": self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓[/green] Saved results to: {output_path}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best performing configuration."""
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")
        return self.results[0]
