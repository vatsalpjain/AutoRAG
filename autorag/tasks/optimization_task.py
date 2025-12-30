"""
Celery task for running optimization in the background.
Wraps existing optimization logic with progress tracking.
"""
from pathlib import Path
from typing import Dict, Any

from autorag.tasks.celery_app import app
from autorag.tasks.progress import ProgressTracker
from autorag.utils.config import load_config
from autorag.database.supabase import SupabaseConnector
from autorag.rag.pipeline import RAGPipeline
from autorag.synthetic.generator import SyntheticQAGenerator
from autorag.optimization.grid_search import GridSearchOptimizer


@app.task(bind=True, name="autorag.optimize")
def run_optimization(self, config_path: str = "config.yaml", num_experiments: int = None):
    """
    Run RAG optimization as a background task.
    
    This is the Celery task version of the CLI optimize command.
    It wraps the same logic but adds progress tracking.
    
    Args:
        config_path: Path to config.yaml file
        num_experiments: Number of experiments to run (overrides config)
        
    Returns:
        Dict with optimization results and best config
    """
    # Initialize progress tracker with Celery task ID
    tracker = ProgressTracker()
    
    try:
        # ========== STEP 1: LOAD CONFIG (5%) ==========
        tracker.start(task_id=self.request.id, total_configs=9)
        tracker.update(current_step="Loading configuration", percent_complete=5)
        
        config = load_config(config_path)
        actual_experiments = num_experiments if num_experiments else config.optimization.num_experiments
        
        # ========== STEP 2: CONNECT TO DATABASE (10%) ==========
        tracker.update(current_step="Connecting to database", percent_complete=10)
        
        if config.database.type != "supabase":
            raise ValueError(f"Database type '{config.database.type}' not yet supported")
        
        connector = SupabaseConnector(config.database)
        connector.test_connection()
        
        doc_count = connector.count_documents()
        if doc_count == 0:
            raise ValueError(f"No documents found in table '{config.database.table}'")
        
        # ========== STEP 3: FETCH DOCUMENTS (15%) ==========
        tracker.update(current_step="Fetching documents", percent_complete=15)
        
        fetch_limit = min(doc_count, 100)
        documents = connector.fetch_documents(limit=fetch_limit)
        
        # ========== STEP 4: INITIALIZE RAG PIPELINE (20%) ==========
        tracker.update(current_step="Initializing RAG pipeline", percent_complete=20)
        
        pipeline = RAGPipeline(
            groq_api_key=config.api_keys.groq,
            pinecone_api_key=config.api_keys.pinecone,
            pinecone_index=config.api_keys.pinecone_index
        )
        
        # ========== STEP 5: INDEX DOCUMENTS (30%) ==========
        tracker.update(current_step="Indexing documents", percent_complete=25)
        
        stats = pipeline.get_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        # Only index if no vectors exist (skip re-indexing in async mode)
        if vector_count == 0:
            tracker.update(current_step="Embedding and indexing documents", percent_complete=30)
            pipeline.index_documents(documents)
        
        # ========== STEP 6: GENERATE SYNTHETIC Q&A (45%) ==========
        tracker.update(current_step="Generating synthetic Q&A pairs", percent_complete=35)
        
        qa_generator = SyntheticQAGenerator(
            groq_api_key=config.api_keys.groq,
            questions_per_doc=2,
            temperature=0.8
        )
        
        qa_pairs = qa_generator.generate(
            documents=documents,
            target_count=config.optimization.test_questions,
            show_progress=False  # No terminal progress in background
        )
        
        # Save Q&A pairs
        output_file = Path("reports/synthetic_qa.json")
        qa_generator.save_to_file(qa_pairs, output_path=output_file)
        
        tracker.update(current_step="Q&A generation complete", percent_complete=45)
        
        # ========== STEP 7: RUN OPTIMIZATION (45% - 95%) ==========
        strategy = config.optimization.strategy
        
        if strategy == "bayesian":
            tracker.update(current_step="Starting Bayesian optimization (Optuna)", percent_complete=50)
            from autorag.optimization.bayesian import BayesianOptimizer
            
            optimizer = BayesianOptimizer(
                pipeline=pipeline,
                groq_api_key=config.api_keys.groq
            )
            
            tracker.update(
                current_step=f"Running {actual_experiments} Bayesian trials",
                percent_complete=55,
                configs_tested=0
            )
            
            # Run Bayesian optimization
            optimizer.optimize(
                qa_pairs=qa_pairs,
                n_trials=actual_experiments,
                show_progress=False  # No terminal progress in background
            )
        else:
            # Default: Grid search
            tracker.update(current_step="Starting grid search optimization", percent_complete=50)
            
            optimizer = GridSearchOptimizer(
                pipeline=pipeline,
                groq_api_key=config.api_keys.groq
            )
            
            max_configs = actual_experiments if actual_experiments <= 20 else 9
            tracker.update(
                current_step=f"Testing {max_configs} configurations",
                percent_complete=55,
                configs_tested=0
            )
            
            # Run grid search optimization
            optimizer.optimize(
                qa_pairs=qa_pairs,
                max_configs=max_configs,
                show_progress=False  # No terminal progress in background
            )
        
        # Save results
        results_file = Path("reports/optimization_results.json")
        optimizer.save_results(output_path=results_file)
        
        # ========== STEP 8: COMPLETE (100%) ==========
        best_config = optimizer.get_best_config()
        
        tracker.complete(best_config=best_config)
        
        return {
            "status": "completed",
            "best_config": best_config,
            "total_configs_tested": len(optimizer.results),
            "results_file": str(results_file),
            "qa_file": str(output_file)
        }
        
    except Exception as e:
        # Record failure in progress tracker
        tracker.fail(error_message=str(e))
        
        # Re-raise so Celery marks task as failed
        raise
