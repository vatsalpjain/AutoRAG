"""
Progress tracker for async optimization tasks.
Saves/loads progress to JSON file for status reporting.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


# Default progress file location
PROGRESS_FILE = Path("reports/progress.json")


@dataclass
class OptimizationProgress:
    """Tracks the state of an optimization run."""
    
    task_id: str                          # Celery task ID
    status: str                           # pending, running, completed, failed
    current_step: str                     # Current step name (e.g., "Indexing documents")
    percent_complete: int                 # 0-100
    started_at: Optional[str] = None      # ISO timestamp
    completed_at: Optional[str] = None    # ISO timestamp
    error_message: Optional[str] = None   # Error details if failed
    
    # Intermediate results (updated as optimization progresses)
    configs_tested: int = 0               # Number of configs tested so far
    total_configs: int = 0                # Total configs to test
    best_config_so_far: Optional[Dict[str, Any]] = None  # Best config found so far


class ProgressTracker:
    """
    Manages optimization progress state via JSON file.
    Thread-safe for single-writer scenarios (Celery worker writes, CLI reads).
    """
    
    def __init__(self, progress_file: Path = PROGRESS_FILE):
        """
        Initialize progress tracker.
        
        Args:
            progress_file: Path to progress JSON file (default: reports/progress.json)
        """
        self.progress_file = Path(progress_file)
        # Ensure reports directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start(self, task_id: str, total_configs: int = 9) -> OptimizationProgress:
        """
        Initialize a new optimization run.
        
        Args:
            task_id: Celery task ID
            total_configs: Total number of configurations to test
            
        Returns:
            New OptimizationProgress object
        """
        progress = OptimizationProgress(
            task_id=task_id,
            status="running",
            current_step="Starting optimization",
            percent_complete=0,
            started_at=datetime.utcnow().isoformat(),
            total_configs=total_configs
        )
        self._save(progress)
        return progress
    
    def update(
        self,
        current_step: str,
        percent_complete: int,
        configs_tested: int = 0,
        best_config: Optional[Dict[str, Any]] = None
    ):
        """
        Update progress state.
        
        Args:
            current_step: Description of current step
            percent_complete: Percentage complete (0-100)
            configs_tested: Number of configs tested so far
            best_config: Best configuration found so far (optional)
        """
        progress = self.load()
        if progress:
            progress.current_step = current_step
            progress.percent_complete = min(percent_complete, 100)
            progress.configs_tested = configs_tested
            if best_config:
                progress.best_config_so_far = best_config
            self._save(progress)
    
    def complete(self, best_config: Dict[str, Any]):
        """
        Mark optimization as complete.
        
        Args:
            best_config: Final best configuration
        """
        progress = self.load()
        if progress:
            progress.status = "completed"
            progress.current_step = "Optimization complete"
            progress.percent_complete = 100
            progress.completed_at = datetime.utcnow().isoformat()
            progress.best_config_so_far = best_config
            self._save(progress)
    
    def fail(self, error_message: str):
        """
        Mark optimization as failed.
        
        Args:
            error_message: Error description
        """
        progress = self.load()
        if progress:
            progress.status = "failed"
            progress.current_step = "Failed"
            progress.completed_at = datetime.utcnow().isoformat()
            progress.error_message = error_message
            self._save(progress)
    
    def load(self) -> Optional[OptimizationProgress]:
        """
        Load progress from file.
        
        Returns:
            OptimizationProgress object or None if no progress file exists
        """
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return OptimizationProgress(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    
    def _save(self, progress: OptimizationProgress):
        """Save progress to file."""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(progress), f, indent=2, default=str)
    
    def clear(self):
        """Delete progress file (for cleanup)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
