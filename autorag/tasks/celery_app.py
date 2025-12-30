"""
Celery application configuration for AutoRAG.
Sets up Celery with Redis as message broker for background task processing.
"""
import os
from celery import Celery

# Redis connection URL (configurable via environment variable)
# Default: localhost Redis on standard port
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Create Celery application
# - 'autorag.tasks' is the namespace for our tasks
# - broker: where tasks are queued (Redis)
# - backend: where results are stored (also Redis)
app = Celery(
    "autorag.tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["autorag.tasks.optimization_task"]  # Auto-discover tasks from this module
)

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",           # Serialize task arguments as JSON
    accept_content=["json"],          # Only accept JSON content
    result_serializer="json",         # Serialize results as JSON
    timezone="UTC",                   # Use UTC for scheduling
    enable_utc=True,
    
    # Queue settings
    task_default_queue="autorag_tasks",  # Default queue name as requested
    
    # Result backend settings
    result_expires=86400,             # Results expire after 24 hours (in seconds)
    
    # Worker settings
    worker_prefetch_multiplier=1,     # Fetch one task at a time (good for long tasks)
    task_acks_late=True,              # Acknowledge task after completion (safer)
    
    # Retry settings for failed tasks
    task_soft_time_limit=21600,       # Soft limit: 6 hours (warning)
    task_time_limit=25200,            # Hard limit: 7 hours (kill task)
)

# This allows running: celery -A autorag.tasks.celery_app worker
if __name__ == "__main__":
    app.start()
