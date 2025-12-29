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

from autorag.utils.config import load_config
from autorag.database.supabase import SupabaseConnector
from autorag.rag.pipeline import RAGPipeline
from autorag.synthetic.generator import SyntheticQAGenerator
from autorag.optimization.grid_search import GridSearchOptimizer

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
    config_file: Path = typer.Option("config.yaml", "--config", "-c", help="Path to config file")
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
        # Only Supabase supported for now
        if config.database.type != "supabase":
            console.print(f"[bold red]❌ Database type '{config.database.type}' not yet supported[/bold red]")
            console.print("[dim]Currently only Supabase is supported. MongoDB and PostgreSQL coming soon.[/dim]")
            raise typer.Exit(code=1)
        
        # Create connector
        connector = SupabaseConnector(config.database)
        
        # Test connection
        console.print("  Testing connection...", end=" ")
        connector.test_connection()
        console.print("[green]✓[/green] Connected")
        
        # Count documents
        doc_count = connector.count_documents()
        console.print(f"  Total documents in table: [yellow]{doc_count}[/yellow]")
        
        if doc_count == 0:
            console.print("[bold red]❌ No documents found in table[/bold red]")
            console.print(f"[dim]Please add documents to '{config.database.table}' table[/dim]")
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
        pipeline = RAGPipeline(
            groq_api_key=config.api_keys.groq,
            pinecone_api_key=config.api_keys.pinecone,
            pinecone_index=config.api_keys.pinecone_index
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
        # Initialize Q&A generator (no model_name needed - wrapper handles rotation)
        qa_generator = SyntheticQAGenerator(
            groq_api_key=config.api_keys.groq,
            questions_per_doc=2,  # Generate 2 questions per document
            temperature=0.8  # Higher temperature for diverse questions
        )
        
        # Generate Q&A pairs
        target_questions = config.optimization.test_questions
        qa_pairs = qa_generator.generate(
            documents=documents,
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
        console.print(f"    Total documents processed: {stats['total_documents']}")
        console.print(f"    Total questions generated: {stats['total_questions']}")
        if stats['failed_generations'] > 0:
            console.print(f"    Failed generations: [yellow]{stats['failed_generations']}[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Synthetic Q&A generation failed:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    
    console.print("\n" + "─" * 60 + "\n")
    
    # ========== RUN GRID SEARCH OPTIMIZATION ==========
    console.print("[bold cyan]🔍 Running Grid Search Optimization[/bold cyan]")
    
    try:
        # Initialize optimizer with existing pipeline and Groq key for Ragas
        optimizer = GridSearchOptimizer(
            pipeline=pipeline,
            groq_api_key=config.api_keys.groq
        )
        
        # Run optimization (test configurations)
        console.print(f"  Testing multiple RAG configurations with Ragas evaluation...\n")
        results = optimizer.optimize(
            qa_pairs=qa_pairs,
            max_configs=num_experiments if num_experiments <= 20 else 9,
            show_progress=True
        )
        
        # Save results to file
        results_file = Path("reports/optimization_results.json")
        optimizer.save_results(output_path=results_file)
        
        # Show best configuration
        best_config = optimizer.get_best_config()
        console.print(f"\n[bold green]🏆 Optimization Complete![/bold green]")
        console.print(f"  Best config: [cyan]{best_config['config']['name']}[/cyan]")
        console.print(f"  Accuracy (Ragas Aggregate): {best_config['metrics']['accuracy']:.3f}")
        
        # Show Ragas metric breakdown if available
        if best_config['metrics'].get('ragas_answer_relevancy') is not None:
            console.print(f"\n  [bold]Ragas Metrics Breakdown:[/bold]")
            console.print(f"    • Answer Relevancy: {best_config['metrics']['ragas_answer_relevancy']:.3f}")
            console.print(f"    • Faithfulness: {best_config['metrics']['ragas_faithfulness']:.3f}")
            console.print(f"    • Context Precision: {best_config['metrics']['ragas_context_precision']:.3f}")
            console.print(f"    • Context Recall: {best_config['metrics']['ragas_context_recall']:.3f}")
            console.print(f"    • Answer Similarity: {best_config['metrics']['ragas_answer_similarity']:.3f}")
        
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
    console.print("[bold cyan]📋 Optimization Summary[/bold cyan]")
    console.print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    console.print(f"  Configurations tested: [yellow]{metadata.get('total_configs_tested', 0)}[/yellow]")
    
    console.print("\n" + "─" * 80 + "\n")
    
    # ========== DISPLAY RESULTS TABLE ==========
    console.print("[bold cyan]🏆 Top Configurations[/bold cyan]\n")
    
    # Create detailed results table
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Rank", style="dim", width=6, justify="center")
    table.add_column("Configuration", style="cyan")
    table.add_column("Ragas Score", justify="right")
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
    console.print(f"    • Ragas Aggregate Score: [green]{best['metrics']['accuracy']:.3f}[/green]")
    console.print(f"    • Avg Token Usage: [yellow]{best['metrics']['avg_tokens']:.0f}[/yellow] tokens/query")
    console.print(f"    • Avg Latency: [cyan]{best['metrics']['avg_latency_seconds']:.2f}[/cyan] seconds")
    console.print(f"    • Success Rate: [green]{(best['metrics']['successful_queries']/best['metrics']['total_queries'])*100:.0f}%[/green]")
    
    console.print(f"\n  [bold]Overall Score (Ragas Aggregate):[/bold] [bold green]{best['weighted_score']:.3f}[/bold green]")
    
    # ========== SHOW RAGAS METRICS BREAKDOWN ==========
    if best['metrics'].get('ragas_answer_relevancy') is not None:
        console.print(f"\n  [bold]Ragas Metrics Breakdown:[/bold]")
        console.print(f"    • Answer Relevancy (30%): [cyan]{best['metrics']['ragas_answer_relevancy']:.3f}[/cyan]")
        console.print(f"    • Faithfulness (25%): [cyan]{best['metrics']['ragas_faithfulness']:.3f}[/cyan]")
        console.print(f"    • Context Precision (15%): [cyan]{best['metrics']['ragas_context_precision']:.3f}[/cyan]")
        console.print(f"    • Context Recall (15%): [cyan]{best['metrics']['ragas_context_recall']:.3f}[/cyan]")
        console.print(f"    • Answer Similarity (15%): [cyan]{best['metrics']['ragas_answer_similarity']:.3f}[/cyan]")
        console.print(f"\n    [dim]Note: Overall score is the weighted average of these Ragas metrics[/dim]")
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


def _format_ragas_breakdown_html(metrics: Dict[str, Any]) -> str:
    """
    Format Ragas metrics breakdown for HTML report.
    
    Args:
        metrics: Metrics dict containing Ragas scores
        
    Returns:
        HTML string with Ragas breakdown or empty string if not available
    """
    if metrics.get('ragas_answer_relevancy') is not None:
        return f"""
        <p><strong>Ragas Metrics Breakdown:</strong></p>
        <ul>
            <li>Answer Relevancy (30%): {metrics['ragas_answer_relevancy']:.3f}</li>
            <li>Faithfulness (25%): {metrics['ragas_faithfulness']:.3f}</li>
            <li>Context Precision (15%): {metrics['ragas_context_precision']:.3f}</li>
            <li>Context Recall (15%): {metrics['ragas_context_recall']:.3f}</li>
            <li>Answer Similarity (15%): {metrics['ragas_answer_similarity']:.3f}</li>
        </ul>
        <p><em>Note: Overall score is the weighted average of these Ragas metrics</em></p>
        """
    return "<p><em>Individual Ragas metrics not available</em></p>"


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
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 AutoRAG Optimization Report</h1>
        <p>Generated: {metadata.get('timestamp', 'N/A')}</p>
        <p>Configurations Tested: {metadata.get('total_configs_tested', 0)}</p>
    </div>
    
    <div class="metric-card">
        <h2>🏆 Best Configuration</h2>
        <p><strong>Name:</strong> {results_list[0]['config']['name']}</p>
        <p><strong>Parameters:</strong> top_k={results_list[0]['config']['top_k']}, temperature={results_list[0]['config']['temperature']}</p>
        <p><strong>Overall Score (Ragas Aggregate):</strong> <span class="score">{results_list[0]['weighted_score']:.3f}</span></p>
        <p><strong>Ragas Aggregate Score:</strong> {results_list[0]['metrics']['accuracy']:.3f}</p>
        {_format_ragas_breakdown_html(results_list[0]['metrics'])}
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
                    <th>Ragas Score</th>
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
    console.print("[bold cyan]Optimization Status[/bold cyan]\n")
    
    # TODO: Check Celery task status
    # TODO: Display progress bar
    # TODO: Show intermediate results
    
    console.print("[yellow]⚠️  Not yet implemented[/yellow]")


# Entry point for the CLI
if __name__ == "__main__":
    app()
