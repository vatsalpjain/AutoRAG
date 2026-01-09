"""
Grid Search Optimization - Tests multiple RAG configurations to find the best one.
Two-phase optimization: indexing params (expensive) × query params (fast).
"""
import json
import time
import itertools
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from autorag.rag.pipeline import RAGPipeline
from autorag.evaluation.evaluator_factory import get_evaluator

# Initialize Rich console for terminal output
console = Console()


class GridSearchOptimizer:
    """
    Grid search optimizer for RAG configurations.
    Uses two-phase optimization:
      - Outer loop: indexing params (chunk_size, chunk_overlap, embedding_model)
      - Inner loop: query params (top_k, temperature)
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
        Initialize grid search optimizer.
        
        Args:
            llm_provider: LLM provider (groq, openai, openrouter)
            llm_api_key: API key for the LLM provider
            llm_model: Optional model name
            evaluation_method: Evaluation method - 'custom' or 'ragas'
            rag_config: RAG parameter search space from config.yaml
            documents: Documents to index (for re-indexing with different params)
        """
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.documents = documents or []
        self.results = []
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
    
    def _create_pipeline(
        self, 
        chunk_size: int, 
        chunk_overlap: int, 
        embedding_model: str,
        collection_name: str
    ) -> RAGPipeline:
        """Create a new pipeline with specific indexing params."""
        return RAGPipeline(
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name
        )
    
    def _generate_smart_configs(
        self,
        indexing_combos: List[tuple],
        query_combos: List[tuple],
        max_configs: int
    ) -> Dict[tuple, List[tuple]]:
        """
        Generate smart configuration sampling using stratified approach.
        Ensures all indexing configs get at least 1 query test if budget allows.
        
        Returns:
            Dict mapping each indexing combo to list of query combos to test
        """
        import random
        
        n_index = len(indexing_combos)
        n_query = len(query_combos)
        total_possible = n_index * n_query
        
        # If budget covers everything, test all combinations
        if max_configs >= total_possible:
            return {idx: list(query_combos) for idx in indexing_combos}
        
        # If budget >= indexing configs, ensure each gets at least some queries
        if max_configs >= n_index:
            # Calculate how many query tests per indexing config
            base_per_index = max_configs // n_index
            remaining = max_configs % n_index
            
            config_map = {}
            for i, idx_combo in enumerate(indexing_combos):
                # Give base amount + 1 extra to first 'remaining' configs
                n_queries = base_per_index + (1 if i < remaining else 0)
                n_queries = min(n_queries, n_query)  # Cap at available queries
                
                # Randomly sample query configs for diversity
                sampled_queries = random.sample(query_combos, n_queries)
                config_map[idx_combo] = sampled_queries
            
            return config_map
        else:
            # Budget < indexing configs: randomly sample indexing configs
            sampled_indexes = random.sample(indexing_combos, max_configs)
            return {idx: [random.choice(query_combos)] for idx in sampled_indexes}
    
    def optimize(
        self,
        qa_pairs: List[Dict[str, Any]],
        max_configs: int = 27,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run two-phase grid search optimization with smart stratified sampling.
        
        Ensures fair coverage of the search space:
        - All indexing configs get tested if budget allows
        - Query configs are distributed evenly across indexing configs
        - Random sampling within each indexing config for diversity
        
        Args:
            qa_pairs: List of Q&A pairs to test against
            max_configs: Maximum total configurations to test
            show_progress: Whether to show progress bars
            
        Returns:
            List of results sorted by weighted score (best first)
        """
        import random
        random.seed(42)  # Reproducible sampling
        
        # Generate all configuration combinations
        indexing_combos = list(itertools.product(
            self.chunk_sizes, self.chunk_overlaps, self.embedding_models
        ))
        query_combos = list(itertools.product(
            self.top_k_values, self.temperature_values
        ))
        
        # Use smart stratified sampling
        config_map = self._generate_smart_configs(indexing_combos, query_combos, max_configs)
        
        total_configs = sum(len(queries) for queries in config_map.values())
        n_indexing_tested = len(config_map)
        
        if show_progress:
            console.print(f"\n[bold cyan]🔍 Smart Grid Search Optimization[/bold cyan]")
            console.print(f"  Total indexing configs: {len(indexing_combos)} (chunk_size × overlap × embedding)")
            console.print(f"  Total query configs: {len(query_combos)} (top_k × temperature)")
            console.print(f"  Budget: {max_configs} experiments")
            console.print(f"  [green]Strategy: Stratified sampling[/green]")
            console.print(f"    → Testing {n_indexing_tested}/{len(indexing_combos)} indexing configs")
            console.print(f"    → {total_configs} total configurations")
            console.print(f"  Evaluating on {len(qa_pairs)} Q&A pairs\n")
        
        config_count = 0
        
        # Iterate through selected indexing configs
        for idx, (idx_combo) in enumerate(config_map.keys()):
            chunk_size, chunk_overlap, emb_model = idx_combo
            query_list = config_map[idx_combo]
            
            # Create unique collection name for this indexing config
            model_short = emb_model.replace("/", "_").replace("-", "_")
            collection_name = f"autorag_c{chunk_size}_o{chunk_overlap}_{model_short}"
            
            if show_progress:
                console.print(f"\n[bold yellow]📦 Indexing Config {idx+1}/{n_indexing_tested}[/bold yellow]")
                console.print(f"  chunk_size={chunk_size}, overlap={chunk_overlap}, model={emb_model}")
                console.print(f"  Testing {len(query_list)} query variations")
            
            # Create new pipeline with these indexing params
            pipeline = self._create_pipeline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=emb_model,
                collection_name=collection_name
            )
            
            # Clear and re-index documents
            if show_progress:
                console.print("  Clearing index...", end=" ")
            pipeline.clear_index()
            if show_progress:
                console.print("[green]✓[/green]")
            
            if show_progress:
                console.print(f"  Indexing {len(self.documents)} documents...", end=" ")
            pipeline.index_documents(self.documents)
            if show_progress:
                console.print("[green]✓[/green]")
            
            # Test the assigned query configs for this index
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
                disable=not show_progress,
                refresh_per_second=2
            ) as progress:
                
                task = progress.add_task(
                    "Testing query configs...",
                    total=len(query_list)
                )
                
                for top_k, temperature in query_list:
                    # Build full config
                    config = {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "embedding_model": emb_model,
                        "top_k": top_k,
                        "temperature": temperature,
                        "name": f"c{chunk_size}_o{chunk_overlap}_k{top_k}_t{temperature:.1f}"
                    }
                    
                    progress.update(task, description=f"Testing k={top_k}, t={temperature:.1f}...")
                    
                    # Evaluate this configuration
                    result = self._evaluate_config(config, pipeline, qa_pairs)
                    self.results.append(result)
                    
                    config_count += 1
                    progress.update(task, advance=1)
        
        # Sort by weighted score (best first)
        self.results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        if show_progress:
            console.print(f"\n[green]✓[/green] Smart grid search complete!")
            console.print(f"  Tested {config_count} configurations across {n_indexing_tested} indexing setups")
            self._display_results_table()
        
        return self.results
    
    def _evaluate_config(
        self,
        config: Dict[str, Any],
        pipeline: RAGPipeline,
        qa_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a single configuration on all Q&A pairs.
        
        Args:
            config: Full configuration dict
            pipeline: Initialized pipeline to use
            qa_pairs: List of Q&A pairs to test
            
        Returns:
            Result dict with metrics and scores
        """
        successful_queries = 0
        rag_results = []
        
        # Run all queries
        for qa_pair in qa_pairs:
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
        
        # Evaluate using evaluator
        eval_scores = {}
        avg_accuracy = 0.5  # Default fallback
        
        if self.evaluator and successful_queries > 0:
            try:
                all_scores = {
                    "answer_relevancy": [], 
                    "faithfulness": [], 
                    "answer_similarity": [], 
                    "context_recall": []
                }
                
                for qa_pair, rag_result in zip(qa_pairs, rag_results):
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
                "total_queries": len(qa_pairs),
                "answer_relevancy": eval_scores.get("answer_relevancy"),
                "faithfulness": eval_scores.get("faithfulness"),
                "answer_similarity": eval_scores.get("answer_similarity"),
                "context_recall": eval_scores.get("context_recall")
            },
            "weighted_score": avg_accuracy
        }
    
    def _display_results_table(self):
        """Display results in a formatted table."""
        console.print("\n[bold cyan]📊 Optimization Results[/bold cyan]\n")
        
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
        
        # Show best config details
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
                "total_configs_tested": len(self.results),
                "evaluation_method": self.evaluation_method,
                "best_config": self.results[0]["config"] if self.results else None
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
