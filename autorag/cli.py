"""
AutoRAG CLI - Command-line interface for RAG optimization.
"""
import json
import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from typing import Dict, Any

from autorag.utils.config import load_config, DatabaseConfig
from autorag.utils.text_utils import chunk_documents, sample_chunks_for_qa
from autorag.database.supabase import SupabaseConnector
from autorag.database.mongodb import MongoDBConnector
from autorag.database.postgres import PostgreSQLConnector
from autorag.rag.pipeline import RAGPipeline
from autorag.synthetic.generator import SyntheticQAGenerator
from autorag.optimization.grid_search import GridSearchOptimizer


def get_connector(config: DatabaseConfig):
    """
    Factory function to get the appropriate database connector.
    
    Args:
        config: DatabaseConfig with 'type' field
        
    Returns:
        Database connector instance (Supabase, MongoDB, or PostgreSQL)
    """
    if config.type == "supabase":
        return SupabaseConnector(config)
    elif config.type == "mongodb":
        return MongoDBConnector(config)
    elif config.type == "postgresql":
        return PostgreSQLConnector(config)
    else:
        raise ValueError(f"Unsupported database type: {config.type}")


# Initialize Typer app and Rich console for beautiful terminal output
app = typer.Typer(
    name="autorag",
    help="AutoRAG Optimizer - Automatically find the optimal RAG configuration for your database",
    add_completion=False  # Disable shell completion for simplicity
)
console = Console()


@app.command()
def optimize(
    experiments: int = typer.Option(None, "--experiments", "-e", help="Number of experiments to run (overrides config)"),
    config_file: Path = typer.Option("config.yaml", "--config", "-c", help="Path to config file"),
    run_async: bool = typer.Option(False, "--async", help="Run optimization in background (requires Celery worker)")
):
    """
    Run the RAG optimization process.
    
    This will:
    1. Load your configuration
    2. Generate synthetic Q&A pairs
    3. Test multiple RAG configurations
    4. Evaluate each config on accuracy, cost, and latency
    5. Save results for analysis
    """
    console.print(Panel.fit(
        "[bold blue]AutoRAG Optimizer[/bold blue]",
        subtitle="Finding your optimal RAG configuration"
    ))
    
    # ========== LOAD & VALIDATE CONFIG ==========
    try:
        console.print(f"\n📝 Loading configuration from: [cyan]{config_file}[/cyan]")
        config = load_config(config_file)
        console.print("[green]✓[/green] Configuration loaded successfully\n")
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ {e}[/bold red]")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]❌ Configuration Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]❌ Unexpected error loading config:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    # Override num_experiments from CLI flag if provided
    num_experiments = experiments if experiments else config.optimization.num_experiments
    
    # ========== ASYNC MODE: DISPATCH TO CELERY ==========
    if run_async:
        try:
            from autorag.tasks.optimization_task import run_optimization
            
            console.print("\n[bold cyan]🚀 Async Mode Enabled[/bold cyan]")
            console.print("  Dispatching optimization task to Celery worker...\n")
            
            # Dispatch task to Celery (non-blocking)
            task = run_optimization.delay(
                config_path=str(config_file),
                num_experiments=num_experiments
            )
            
            console.print(f"[green]✓[/green] Task dispatched successfully!")
            console.print(f"  Task ID: [cyan]{task.id}[/cyan]")
            console.print("\n[bold]Next Steps:[/bold]")
            console.print("  1. Check progress: [cyan]autorag status[/cyan]")
            console.print("  2. View results when complete: [cyan]autorag results[/cyan]")
            console.print("\n[dim]Note: Ensure Celery worker is running:[/dim]")
            console.print("  [dim]celery -A autorag.tasks.celery_app worker -Q autorag_tasks --loglevel=info[/dim]")
            return  # Exit immediately, worker handles the rest
            
        except ImportError as e:
            console.print(f"[bold red]❌ Celery not available:[/bold red] {e}")
            console.print("[dim]Install with: pip install celery redis[/dim]")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]❌ Failed to dispatch async task:[/bold red] {e}")
            console.print("[dim]Is Redis running? Try: docker-compose up -d redis[/dim]")
            raise typer.Exit(code=1)
    
    # ========== DISPLAY CONFIGURATION ==========
    console.print("[bold cyan]📋 Configuration Summary[/bold cyan]")
    
    # Database info
    console.print(f"  Database: [yellow]{config.database.type}[/yellow]")
    if config.database.type == "supabase":
        console.print(f"    - URL: {config.database.url}")
        console.print(f"    - Table: {config.database.table}")
    elif config.database.type == "mongodb":
        console.print(f"    - Database: {config.database.database}")
        console.print(f"    - Collection: {config.database.collection}")
    elif config.database.type == "postgresql":
        console.print(f"    - Host: {config.database.host}:{config.database.port}")
        console.print(f"    - Database: {config.database.database}")
    
    # API keys (masked)
    console.print(f"\n  API Keys:")
    console.print(f"    - Groq: [green]✓[/green] {config.api_keys.groq[:8]}...")
    console.print(f"    - Pinecone: [green]✓[/green] {config.api_keys.pinecone[:8]}...")
    
    # Optimization settings
    console.print(f"\n  Optimization:")
    console.print(f"    - Experiments: [yellow]{num_experiments}[/yellow]")
    console.print(f"    - Test Questions: [yellow]{config.optimization.test_questions}[/yellow]")
    
    console.print("\n" + "─" * 60 + "\n")
    
    # ========== CONNECT TO DATABASE ==========
    console.print("[bold cyan]🔌 Connecting to Database[/bold cyan]")
    
    try:
        # Create connector using factory function (supports all 3 database types)
        connector = get_connector(config.database)
        
        # Test connection
        console.print("  Testing connection...", end=" ")
        connector.test_connection()
        console.print("[green]✓[/green] Connected")
        
        # Count documents
        doc_count = connector.count_documents()
        console.print(f"  Total documents: [yellow]{doc_count}[/yellow]")
        
        if doc_count == 0:
            console.print("[bold red]❌ No documents found[/bold red]")
            console.print("[dim]Please add documents to your database[/dim]")
            raise typer.Exit(code=1)
        
    except Exception as e:
        console.print(f"[bold red]❌ Database connection failed:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    # ========== FETCH DOCUMENTS ==========
    console.print("\n[bold cyan]📚 Fetching Documents[/bold cyan]")
    
    try:
        # Fetch sample documents (limit to avoid overwhelming memory)
        fetch_limit = min(doc_count, 100)
        console.print(f"  Fetching {fetch_limit} documents...", end=" ")
        
        documents = connector.fetch_documents(limit=fetch_limit)
        console.print(f"[green]✓[/green] Fetched {len(documents)} documents")
        
        # Show sample document info
        if documents:
            sample_doc = documents[0]
            console.print(f"\n  [dim]Sample document:[/dim]")
            console.print(f"    ID: {sample_doc['id']}")
            console.print(f"    Text length: {len(sample_doc['text'])} characters")
            console.print(f"    Text preview: {sample_doc['text'][:100]}...")
            if sample_doc['metadata']:
                console.print(f"    Metadata fields: {', '.join(sample_doc['metadata'].keys())}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to fetch documents:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    console.print("\n" + "─" * 60 + "\n")
    
    # ========== INITIALIZE RAG PIPELINE ==========
    console.print("[bold cyan]🤖 Initializing RAG Pipeline[/bold cyan]")
    
    try:
        console.print("  Creating RAG pipeline...", end=" ")
        # Get LLM config
        llm_provider = config.llm.provider
        llm_api_key = getattr(config.api_keys, llm_provider)
        llm_model = config.llm.model
        
        pipeline = RAGPipeline(
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            pinecone_api_key=config.api_keys.pinecone,
            pinecone_index=config.api_keys.pinecone_index,
            llm_model=llm_model
        )
        console.print("[green]✓[/green] Pipeline ready")
        
        # Check if index already has vectors
        stats = pipeline.get_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        console.print(f"  Vectors in Pinecone: [yellow]{vector_count}[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to initialize RAG pipeline:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    # ========== INDEX DOCUMENTS ==========
    console.print("\n[bold cyan]📊 Indexing Documents[/bold cyan]")
    
    try:
        # Ask user if they want to re-index (if vectors already exist)
        skip_indexing = False  # Initialize default value
        
        if vector_count > 0:
            console.print(f"  [yellow]⚠️  Index already contains {vector_count} vectors[/yellow]")
            reindex = typer.confirm("  Do you want to clear and re-index?", default=False)
            if reindex:
                console.print("  Clearing existing vectors...", end=" ")
                pipeline.clear_index()
                console.print("[green]✓[/green] Cleared")
                skip_indexing = False  # Proceed with indexing after clearing
            else:
                console.print("  [dim]Skipping indexing, using existing vectors[/dim]")
                skip_indexing = True  # Skip indexing, use existing vectors
        
        if not skip_indexing:
            console.print(f"  Embedding and indexing {len(documents)} documents...", end=" ")
            pipeline.index_documents(documents)
            console.print("[green]✓[/green] Indexed")
            
            # Verify indexing
            new_stats = pipeline.get_index_stats()
            new_count = new_stats.get('total_vector_count', 0)
            console.print(f"  Total vectors in index: [yellow]{new_count}[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to index documents:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    # ========== TEST RAG QUERY ==========
    console.print("\n[bold cyan]🧪 Testing RAG Pipeline[/bold cyan]")
    
    try:
        # Test with a simple query
        test_query = "What is the main topic discussed in these documents?"
        console.print(f"  Query: [dim]{test_query}[/dim]\n")
        
        console.print("  Retrieving relevant documents...", end=" ")
        result = pipeline.query(test_query, top_k=3)
        console.print("[green]✓[/green] Done\n")
        
        # Display answer
        console.print("  [bold]Answer:[/bold]")
        console.print(f"  {result['answer']}\n")
        
        # Display sources
        console.print("  [bold]Sources:[/bold]")
        for i, source in enumerate(result['sources'], 1):
            console.print(f"    {i}. Score: {source['score']:.3f} | {source['text']}")
        
    except Exception as e:
        console.print(f"[bold red]❌ RAG query failed:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    console.print("\n" + "─" * 60 + "\n")
    
    # ========== GENERATE SYNTHETIC Q&A PAIRS ==========
    console.print("[bold cyan]📝 Generating Synthetic Q&A Pairs[/bold cyan]")
    
    try:
        # Step 1: Chunk documents for diverse Q&A coverage
        console.print("  Chunking documents for diversity...", end=" ")
        all_chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        console.print(f"[green]✓[/green] Created {len(all_chunks)} chunks")
        
        # Step 2: Randomly sample chunks for Q&A (ensures diversity)
        target_questions = config.optimization.test_questions
        sampled_chunks = sample_chunks_for_qa(all_chunks, target_questions, questions_per_chunk=1)
        console.print(f"  Randomly sampled {len(sampled_chunks)} chunks for Q&A generation")
        
        # Step 3: Initialize Q&A generator
        qa_generator = SyntheticQAGenerator(
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            questions_per_doc=1,  # 1 question per chunk (already sampled)
            temperature=0.8  # Higher temperature for diverse questions
        )
        
        # Step 4: Generate Q&A pairs from sampled chunks
        qa_pairs = qa_generator.generate(
            documents=sampled_chunks,  # Pass chunks as "documents"
            target_count=target_questions,
            show_progress=True
        )
        
        # Save to file
        output_file = Path("reports/synthetic_qa.json")
        qa_generator.save_to_file(qa_pairs, output_path=output_file)
        
        # Show sample Q&A pair
        if qa_pairs:
            console.print(f"\n  [dim]Sample Q&A pair:[/dim]")
            sample = qa_pairs[0]
            console.print(f"    Q: {sample['question']}")
            console.print(f"    A: {sample['answer'][:100]}...")
        
        # Show statistics
        stats = qa_generator.get_statistics()
        console.print(f"\n  [bold]Generation Statistics:[/bold]")
        console.print(f"    Total chunks used: {stats['total_documents']}")
        console.print(f"    Total questions generated: {stats['total_questions']}")
        if stats['failed_generations'] > 0:
            console.print(f"    Failed generations: [yellow]{stats['failed_generations']}[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Synthetic Q&A generation failed:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    console.print("\n" + "─" * 60 + "\n")
    
    # ========== RUN OPTIMIZATION (Grid or Bayesian based on config) ==========
    strategy = config.optimization.strategy
    evaluation_method = config.evaluation.method
    
    try:
        if strategy == "bayesian":
            console.print("[bold cyan]🧠 Running Bayesian Optimization (Optuna)[/bold cyan]")
            console.print(f"  Evaluation Method: [yellow]{evaluation_method.upper()}[/yellow]")
            from autorag.optimization.bayesian import BayesianOptimizer
            
            optimizer = BayesianOptimizer(
                pipeline=pipeline,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                evaluation_method=evaluation_method
            )
            
            console.print(f"  Running {num_experiments} trials with intelligent sampling...\n")
            optimizer.optimize(
                qa_pairs=qa_pairs,
                n_trials=num_experiments,
                show_progress=True
            )
            
        else:  # Default: grid search
            console.print("[bold cyan]🔍 Running Grid Search Optimization[/bold cyan]")
            console.print(f"  Evaluation Method: [yellow]{evaluation_method.upper()}[/yellow]")
            
            optimizer = GridSearchOptimizer(
                pipeline=pipeline,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                evaluation_method=evaluation_method
            )
            
            console.print("  Testing multiple RAG configurations...\n")
            optimizer.optimize(
                qa_pairs=qa_pairs,
                max_configs=num_experiments if num_experiments <= 20 else 9,
                show_progress=True
            )
        
        # Save results to file (common for both strategies)
        results_file = Path("reports/optimization_results.json")
        optimizer.save_results(output_path=results_file)
        
        # Show best configuration
        best_config = optimizer.get_best_config()
        actual_eval_method = optimizer.evaluation_method
        console.print(f"\n[bold green]🏆 Optimization Complete![/bold green]")
        console.print(f"  Evaluation Method: [cyan]{actual_eval_method.upper()}[/cyan]")
        console.print(f"  Best config: [cyan]{best_config['config']['name']}[/cyan]")
        console.print(f"  Accuracy ({actual_eval_method.capitalize()} Aggregate): {best_config['metrics']['accuracy']:.3f}")
        
        # Show metric breakdown if available
        if best_config['metrics'].get('answer_relevancy') is not None:
            console.print(f"\n  [bold]{actual_eval_method.capitalize()} Metrics Breakdown:[/bold]")
            console.print(f"    • Answer Relevancy: {best_config['metrics']['answer_relevancy']:.3f}")
            console.print(f"    • Faithfulness: {best_config['metrics']['faithfulness']:.3f}")
            if best_config['metrics'].get('answer_similarity') is not None:
                console.print(f"    • Answer Similarity: {best_config['metrics']['answer_similarity']:.3f}")
            if best_config['metrics'].get('context_recall') is not None:
                console.print(f"    • Context Recall: {best_config['metrics']['context_recall']:.3f}")
        
        console.print(f"\n  Avg Tokens: {best_config['metrics']['avg_tokens']:.0f}")
        console.print(f"  Latency: {best_config['metrics']['avg_latency_seconds']:.2f}s")
        console.print(f"  Weighted Score: [green]{best_config['weighted_score']:.3f}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Optimization failed:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    console.print("\n" + "─" * 60 + "\n")
    
    console.print("[green]✅ AutoRAG Optimization Complete![/green]")
    console.print(f"[dim]Results saved to: {results_file}[/dim]")
    console.print(f"[dim]Q&A pairs saved to: {output_file}[/dim]")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Review results: autorag results")
    console.print("  2. Deploy best config in your production RAG system")


@app.command()
def results(
    show_report: bool = typer.Option(False, "--show-report", help="Open HTML report in browser"),
    results_file: Path = typer.Option("reports/optimization_results.json", "--file", "-f", help="Path to results file")
):
    """
    Display optimization results.
    
    Shows:
    - Best performing configurations
    - Accuracy, cost, and latency metrics
    - Recommended configuration based on priorities
    """
    console.print(Panel.fit(
        "[bold green]📊 AutoRAG Optimization Results[/bold green]",
        subtitle="Analysis of tested configurations"
    ))
    
    # ========== LOAD RESULTS FILE ==========
    try:
        console.print(f"\n📂 Loading results from: [cyan]{results_file}[/cyan]")
        
        if not results_file.exists():
            console.print(f"[bold red]❌ Results file not found: {results_file}[/bold red]")
            console.print("[dim]Run 'autorag optimize' first to generate results.[/dim]")
            raise typer.Exit(code=1)
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results_list = data.get("results", [])
        metadata = data.get("metadata", {})
        
        if not results_list:
            console.print("[bold red]❌ No results found in file[/bold red]")
            raise typer.Exit(code=1)
        
        console.print("[green]✓[/green] Results loaded successfully\n")
        
    except json.JSONDecodeError as e:
        console.print(f"[bold red]❌ Invalid JSON in results file:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load results:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    # ========== DISPLAY METADATA ==========
    eval_method = metadata.get('evaluation_method', 'custom')
    console.print("[bold cyan]📋 Optimization Summary[/bold cyan]")
    console.print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    console.print(f"  Configurations tested: [yellow]{metadata.get('total_configs_tested', 0)}[/yellow]")
    console.print(f"  Evaluation Method: [cyan]{eval_method.upper()}[/cyan]")
    
    console.print("\n" + "─" * 80 + "\n")
    
    # ========== DISPLAY RESULTS TABLE ==========
    console.print("[bold cyan]🏆 Top Configurations[/bold cyan]\n")
    
    # Create detailed results table
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Rank", style="dim", width=6, justify="center")
    table.add_column("Configuration", style="cyan")
    table.add_column(f"{eval_method.capitalize()} Score", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Overall\nScore", justify="right", style="bold green")
    
    # Add all results to table
    for i, result in enumerate(results_list, 1):
        config = result["config"]
        metrics = result["metrics"]
        
        # Calculate success rate
        success_rate = (metrics["successful_queries"] / metrics["total_queries"]) * 100
        
        # Highlight best config
        rank_style = "bold yellow" if i == 1 else "dim"
        
        table.add_row(
            f"#{i}" if i > 1 else "🥇",
            f"{config['name']}\n(k={config['top_k']}, t={config['temperature']})",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['avg_tokens']:.0f}",
            f"{metrics['avg_latency_seconds']:.2f}",
            f"{success_rate:.0f}%",
            f"{result['weighted_score']:.3f}",
            style=rank_style if i == 1 else None
        )
    
    console.print(table)
    
    # ========== DISPLAY BEST CONFIGURATION DETAILS ==========
    console.print("\n" + "─" * 80 + "\n")
    
    best = results_list[0]
    console.print("[bold green]🎯 Recommended Configuration[/bold green]\n")
    
    console.print(f"  [bold]Configuration Name:[/bold] {best['config']['name']}")
    console.print(f"  [bold]Parameters:[/bold]")
    console.print(f"    • top_k: [cyan]{best['config']['top_k']}[/cyan] documents")
    console.print(f"    • temperature: [cyan]{best['config']['temperature']}[/cyan]")
    
    console.print(f"\n  [bold]Performance Metrics:[/bold]")
    console.print(f"    • {eval_method.capitalize()} Aggregate Score: [green]{best['metrics']['accuracy']:.3f}[/green]")
    console.print(f"    • Avg Token Usage: [yellow]{best['metrics']['avg_tokens']:.0f}[/yellow] tokens/query")
    console.print(f"    • Avg Latency: [cyan]{best['metrics']['avg_latency_seconds']:.2f}[/cyan] seconds")
    console.print(f"    • Success Rate: [green]{(best['metrics']['successful_queries']/best['metrics']['total_queries'])*100:.0f}%[/green]")
    
    console.print(f"\n  [bold]Overall Score ({eval_method.capitalize()} Aggregate):[/bold] [bold green]{best['weighted_score']:.3f}[/bold green]")
    
    # ========== SHOW METRICS BREAKDOWN ==========
    if best['metrics'].get('answer_relevancy') is not None:
        console.print(f"\n  [bold]{eval_method.capitalize()} Metrics Breakdown:[/bold]")
        console.print(f"    • Answer Relevancy: [cyan]{best['metrics']['answer_relevancy']:.3f}[/cyan]")
        console.print(f"    • Faithfulness: [cyan]{best['metrics']['faithfulness']:.3f}[/cyan]")
        if best['metrics'].get('answer_similarity') is not None:
            console.print(f"    • Answer Similarity: [cyan]{best['metrics']['answer_similarity']:.3f}[/cyan]")
        if best['metrics'].get('context_recall') is not None:
            console.print(f"    • Context Recall: [cyan]{best['metrics']['context_recall']:.3f}[/cyan]")
        console.print(f"\n    [dim]Note: Overall score is the weighted average of {eval_method} metrics[/dim]")
    else:
        console.print(f"\n  [dim]Individual Ragas metrics not available (fallback mode used)[/dim]")
    
    # ========== COMPARISON WITH WORST ==========
    if len(results_list) > 1:
        worst = results_list[-1]
        console.print("\n" + "─" * 80 + "\n")
        console.print("[bold cyan]📈 Improvement over Worst Config[/bold cyan]\n")
        
        acc_improvement = ((best['metrics']['accuracy'] - worst['metrics']['accuracy']) / worst['metrics']['accuracy']) * 100
        token_reduction = ((worst['metrics']['avg_tokens'] - best['metrics']['avg_tokens']) / worst['metrics']['avg_tokens']) * 100
        latency_improvement = ((worst['metrics']['avg_latency_seconds'] - best['metrics']['avg_latency_seconds']) / worst['metrics']['avg_latency_seconds']) * 100
        
        console.print(f"  Accuracy: [green]{acc_improvement:+.1f}%[/green]")
        console.print(f"  Token Usage: [green]{token_reduction:+.1f}%[/green] reduction")
        console.print(f"  Latency: [green]{latency_improvement:+.1f}%[/green] improvement")
    
    # ========== GENERATE HTML REPORT (OPTIONAL) ==========
    if show_report:
        console.print("\n" + "─" * 80 + "\n")
        console.print("[bold cyan]📄 Generating HTML Report[/bold cyan]\n")
        
        try:
            html_path = _generate_html_report(data, results_file.parent)
            console.print(f"[green]✓[/green] HTML report generated: {html_path}")
            
            # Open in browser
            import webbrowser
            webbrowser.open(f"file://{html_path.absolute()}")
            console.print("[green]✓[/green] Opened in default browser")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to generate HTML report: {e}[/yellow]")
    
    console.print("\n" + "─" * 80 + "\n")
    console.print("[bold]💡 Next Steps:[/bold]")
    console.print(f"  1. Use the best config (k={best['config']['top_k']}, temp={best['config']['temperature']}) in your RAG system")
    console.print("  2. Run 'autorag optimize' again with different parameters to explore more configs")
    console.print("  3. Use '--show-report' flag to see detailed HTML report")


def _format_metrics_breakdown_html(metrics: Dict[str, Any], eval_method: str) -> str:
    """
    Format metrics breakdown for HTML report.
    
    Args:
        metrics: Metrics dict containing scores
        eval_method: Evaluation method name
        
    Returns:
        HTML string with metrics breakdown or empty string if not available
    """
    if metrics.get('answer_relevancy') is not None:
        breakdown = f"""
        <p><strong>{eval_method.capitalize()} Metrics Breakdown:</strong></p>
        <ul>
            <li>Answer Relevancy: {metrics['answer_relevancy']:.3f}</li>
            <li>Faithfulness: {metrics['faithfulness']:.3f}</li>"""
        if metrics.get('answer_similarity') is not None:
            breakdown += f"\n            <li>Answer Similarity: {metrics['answer_similarity']:.3f}</li>"
        if metrics.get('context_recall') is not None:
            breakdown += f"\n            <li>Context Recall: {metrics['context_recall']:.3f}</li>"
        breakdown += f"""
        </ul>
        <p><em>Note: Overall score is the weighted average of {eval_method} metrics</em></p>
        """
        return breakdown
    return "<p><em>Individual metrics not available</em></p>"


def _generate_html_report(data: Dict[str, Any], output_dir: Path) -> Path:
    """
    Generate an HTML report from optimization results.
    
    Args:
        data: Results data dict
        output_dir: Directory to save HTML report
        
    Returns:
        Path to generated HTML file
    """
    html_path = output_dir / "optimization_report.html"
    
    results_list = data.get("results", [])
    metadata = data.get("metadata", {})
    eval_method = metadata.get('evaluation_method', 'custom')
    
    # Simple HTML template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AutoRAG Optimization Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .best-config {{
            background: #d4edda !important;
            font-weight: bold;
        }}
        .score {{
            font-size: 24px;
            color: #28a745;
            font-weight: bold;
        }}
        .eval-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 AutoRAG Optimization Report</h1>
        <p>Generated: {metadata.get('timestamp', 'N/A')}</p>
        <p>Configurations Tested: {metadata.get('total_configs_tested', 0)}</p>
        <div class="eval-badge">📊 Evaluation Method: {eval_method.upper()}</div>
    </div>
    
    <div class="metric-card">
        <h2>🏆 Best Configuration</h2>
        <p><strong>Name:</strong> {results_list[0]['config']['name']}</p>
        <p><strong>Parameters:</strong> top_k={results_list[0]['config']['top_k']}, temperature={results_list[0]['config']['temperature']}</p>
        <p><strong>Overall Score ({eval_method.capitalize()} Aggregate):</strong> <span class="score">{results_list[0]['weighted_score']:.3f}</span></p>
        <p><strong>{eval_method.capitalize()} Aggregate Score:</strong> {results_list[0]['metrics']['accuracy']:.3f}</p>
        {_format_metrics_breakdown_html(results_list[0]['metrics'], eval_method)}
        <p><strong>Avg Tokens:</strong> {results_list[0]['metrics']['avg_tokens']:.0f}</p>
        <p><strong>Latency:</strong> {results_list[0]['metrics']['avg_latency_seconds']:.2f}s</p>
    </div>
    
    <div class="metric-card">
        <h2>📊 All Configurations</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Config</th>
                    <th>{eval_method.capitalize()} Score</th>
                    <th>Avg Tokens</th>
                    <th>Latency (s)</th>
                    <th>Overall Score</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add table rows
    for i, result in enumerate(results_list, 1):
        row_class = "best-config" if i == 1 else ""
        html_content += f"""
                <tr class="{row_class}">
                    <td>#{i}</td>
                    <td>{result['config']['name']}<br><small>k={result['config']['top_k']}, t={result['config']['temperature']}</small></td>
                    <td>{result['metrics']['accuracy']:.3f}</td>
                    <td>{result['metrics']['avg_tokens']:.0f}</td>
                    <td>{result['metrics']['avg_latency_seconds']:.2f}</td>
                    <td><strong>{result['weighted_score']:.3f}</strong></td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="metric-card">
        <h2>💡 Recommendations</h2>
        <ul>
            <li>Deploy the best configuration in your production RAG system</li>
            <li>Monitor real-world performance and adjust if needed</li>
            <li>Re-run optimization periodically as your data evolves</li>
        </ul>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path


@app.command()
def status(
    config_file: Path = typer.Option("config.yaml", "--config", "-c", help="Path to config file")
):
    """
    Check the status of running optimization.
    
    Shows:
    - Current progress (experiments completed)
    - Estimated time remaining
    - Best configuration so far
    """
    from autorag.tasks.progress import ProgressTracker
    from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
    
    console.print(Panel.fit(
        "[bold cyan]📊 Optimization Status[/bold cyan]",
        subtitle="Background task progress"
    ))
    
    # Load progress from file
    tracker = ProgressTracker()
    progress = tracker.load()
    
    if progress is None:
        console.print("\n[yellow]⚠️  No optimization in progress[/yellow]")
        console.print("[dim]Run 'autorag optimize --async' to start a background optimization[/dim]")
        return
    
    # Display task info
    console.print(f"\n[bold]Task ID:[/bold] [cyan]{progress.task_id}[/cyan]")
    console.print(f"[bold]Status:[/bold] ", end="")
    
    # Color-coded status
    if progress.status == "running":
        console.print("[blue]🔄 Running[/blue]")
    elif progress.status == "completed":
        console.print("[green]✅ Completed[/green]")
    elif progress.status == "failed":
        console.print("[red]❌ Failed[/red]")
    else:
        console.print(f"[yellow]{progress.status}[/yellow]")
    
    # Timestamps
    if progress.started_at:
        console.print(f"[bold]Started:[/bold] {progress.started_at}")
    if progress.completed_at:
        console.print(f"[bold]Completed:[/bold] {progress.completed_at}")
    
    console.print("\n" + "─" * 60 + "\n")
    
    # Progress bar
    console.print(f"[bold]Current Step:[/bold] {progress.current_step}")
    console.print(f"[bold]Progress:[/bold] {progress.percent_complete}%")
    
    # Visual progress bar using Rich
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress_bar:
        task = progress_bar.add_task("Optimization", total=100, completed=progress.percent_complete)
        progress_bar.refresh()
    
    # Configs tested
    if progress.total_configs > 0:
        console.print(f"\n[bold]Configurations:[/bold] {progress.configs_tested}/{progress.total_configs} tested")
    
    # Best config so far
    if progress.best_config_so_far:
        console.print("\n" + "─" * 60)
        console.print("\n[bold green]🏆 Best Configuration So Far[/bold green]")
        config = progress.best_config_so_far
        
        if isinstance(config, dict):
            if "config" in config:
                console.print(f"  Name: [cyan]{config['config'].get('name', 'N/A')}[/cyan]")
                console.print(f"  top_k: {config['config'].get('top_k', 'N/A')}")
                console.print(f"  temperature: {config['config'].get('temperature', 'N/A')}")
            if "metrics" in config:
                console.print(f"  Accuracy: [green]{config['metrics'].get('accuracy', 0):.3f}[/green]")
            if "weighted_score" in config:
                console.print(f"  Weighted Score: [green]{config.get('weighted_score', 0):.3f}[/green]")
    
    # Error message if failed
    if progress.status == "failed" and progress.error_message:
        console.print("\n" + "─" * 60)
        console.print("\n[bold red]Error Details:[/bold red]")
        console.print(f"  {progress.error_message}")
    
    # Next steps
    console.print("\n" + "─" * 60 + "\n")
    if progress.status == "running":
        console.print("[bold]💡 Tip:[/bold] Run 'autorag status' again to see updated progress")
    elif progress.status == "completed":
        console.print("[bold]Next:[/bold] Run 'autorag results' to see full results")


# Entry point for the CLI
if __name__ == "__main__":
    app()
