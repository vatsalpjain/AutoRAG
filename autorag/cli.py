"""
AutoRAG CLI - Command-line interface for RAG optimization.
"""
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
        if vector_count > 0:
            console.print(f"  [yellow]⚠️  Index already contains {vector_count} vectors[/yellow]")
            reindex = typer.confirm("  Do you want to clear and re-index?", default=False)
            if reindex:
                console.print("  Clearing existing vectors...", end=" ")
                pipeline.clear_index()
                console.print("[green]✓[/green] Cleared")
            else:
                console.print("  [dim]Skipping indexing, using existing vectors[/dim]")
                skip_indexing = True
        else:
            skip_indexing = False
        
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
    
    # TODO: Generate synthetic Q&A pairs
    # TODO: Run optimization experiments
    # TODO: Evaluate configurations
    # TODO: Save results
    
    console.print("[green]✅ RAG pipeline is working![/green]")
    console.print("[dim]Next steps: Implement synthetic Q&A generation for optimization[/dim]")


@app.command()
def results(
    show_report: bool = typer.Option(False, "--show-report", help="Open HTML report in browser"),
    config_file: Path = typer.Option("config.yaml", "--config", "-c", help="Path to config file")
):
    """
    Display optimization results.
    
    Shows:
    - Best performing configurations
    - Accuracy, cost, and latency metrics
    - Recommended configuration based on priorities
    """
    console.print("[bold green]Optimization Results[/bold green]\n")
    
    # TODO: Load results from file
    # TODO: Display results table
    # TODO: Show Pareto frontier
    # TODO: Open HTML report if requested
    
    console.print("[yellow]⚠️  Not yet implemented[/yellow]")


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
