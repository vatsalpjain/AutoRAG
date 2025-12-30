# AutoRAG Project - AI Agent Instructions

## 🚨 CRITICAL WORKFLOW RULES (Read First!)

**This is a resume project - the developer is in complete control.**

1. **Write code with comments** (concise, not verbose)
2. **Explain everything you do** - no silent changes
3. **Ask before making decisions** about structure, tech choices, or implementation approaches
4. **Go step-by-step** - do ONE thing, explain it, wait for approval before proceeding
5. **User approval required** - never assume the next step
6. **No need to create any md to explain**- never create md just talk in chat

**Example:** "I've implemented X. Should I proceed with Y, or would you like to review/modify X first?"

---

## Project Overview
AutoRAG is a **pip-installable CLI tool** that automates RAG hyperparameter optimization. Users connect their database, run optimization, and get the best RAG config in 4-6 hours. NOT a web platform.

**📖 Full Architecture & Dataflow:** Read [AutoRAG_CLI.md](../AutoRAG_CLI.md) for complete spec, build plan, and tech decisions.

## Architecture (Quick Reference)
- **CLI** (`autorag/cli.py`): Typer + Rich formatting (`optimize`, `init`, `results`, `status`)
- **RAG Pipeline** (`autorag/rag/`): HuggingFace embeddings (384-dim) → Pinecone (cosine) → Groq LLM
- **Databases** (`autorag/database/`): Supabase ✓ | MongoDB (todo) | PostgreSQL (todo)
- **Config** (`autorag/utils/config.py`): Pydantic validation, YAML-based
- **Optimization** (`autorag/optimization/`): Grid search → Bayesian (Optuna later)
- **Evaluation** (`autorag/evaluation/`): Ragas metrics (accuracy, cost, latency)
- **Async** (Week 3): Celery + Redis for background processing

## Code Conventions (Project-Specific)

### Import Order
```python
# Standard library
from pathlib import Path
from typing import Dict, Any

# Third-party (alphabetical)
from rich.console import Console
import typer

# Local (alphabetical)
from autorag.utils.config import load_config
```

### CLI Error Handling
- Catch specific exceptions, use Rich formatting: `[red]`, `[green]`, `[cyan]`
- Always `raise typer.Exit(code=1)` on fatal errors
- Mask API keys in output: `config.api_keys.groq[:8]...`

### Document Format (All DB Connectors)
```python
{
    "id": str,           # Required
    "text": str,         # Required  
    "metadata": dict     # All other fields
}
```

### Batch Processing Rules
- Embeddings: Use `embed_batch()` with `batch_size=32` (NOT loops)
- Pinecone: Upsert max 100 vectors/batch
- Progress bars: Show for batches > 10 items
- Metadata limit: 1000 chars max (Pinecone constraint)

## Quick Reference

### Key Files
- [AutoRAG_CLI.md](../AutoRAG_CLI.md): Full spec, build plan, 4-week timeline
- [config.py](../autorag/utils/config.py): Pydantic models & validation
- [pipeline.py](../autorag/rag/pipeline.py): RAG flow example
- [cli.py](../autorag/cli.py): CLI patterns & Rich formatting

### External Services
- **Groq**: LLM (llama-3.3-70b-versatile, temp=0.7 default)
- **Pinecone**: Serverless vectors (us-east-1, cosine similarity)
- **Supabase**: Database (fetch_documents limit=100)
- **HuggingFace**: Embeddings (all-MiniLM-L6-v2, cached locally)

### Not Yet Implemented
- MongoDB & PostgreSQL connectors (`autorag/database/`)
- Synthetic Q&A generation (`autorag/synthetic/`)
- Grid search optimization (`autorag/optimization/`)
- Ragas evaluation (`autorag/evaluation/`)
- Celery tasks for async processing

### Anti-Patterns
- ❌ Loop through API calls → ✅ Use batch methods
- ❌ Hardcode API keys → ✅ Load from config.yaml
- ❌ Generic exception catching → ✅ Specific exceptions with context
- ❌ Create CLI commands without Typer docstrings

