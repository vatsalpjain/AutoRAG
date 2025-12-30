"""
AutoRAG Tasks Module - Celery background tasks for optimization.

This module provides:
- Celery app configuration (celery_app.py)
- Progress tracking (progress.py)
- Optimization background task (optimization_task.py)

Usage:
    # Start worker
    celery -A autorag.tasks.celery_app worker --loglevel=info -Q autorag_tasks
    
    # Run async optimization
    autorag optimize --async
    
    # Check status
    autorag status
"""

# Export Celery app for easy access
from autorag.tasks.celery_app import app as celery_app

# Export progress tracker
from autorag.tasks.progress import ProgressTracker, OptimizationProgress

# Export optimization task
from autorag.tasks.optimization_task import run_optimization

__all__ = [
    "celery_app",
    "ProgressTracker",
    "OptimizationProgress",
    "run_optimization",
]
