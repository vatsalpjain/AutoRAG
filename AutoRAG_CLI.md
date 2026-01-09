# AutoRAG Optimizer - CLI Package Project

AutoRAG Optimizer is a self-hosted tool that automatically finds the optimal RAG (Retrieval-Augmented Generation) configuration for any database. Companies waste weeks manually testing RAG settings like chunk size, embedding models, and retrieval strategies without knowing what's actually best. AutoRAG solves this by automating the entire optimization process.

Users install the tool via pip, connect their database (Supabase, MongoDB, or PostgreSQL), and provide their API keys. The system then generates synthetic test questions from their documents using LLMs, eliminating the need for manual labeling. It intelligently searches through the configuration space using Bayesian optimization, testing 20-30 different RAG setups instead of all 1000+ possible combinations.

Each configuration is evaluated across accuracy metrics (using RAGAS-like evaluation). The optimization uses a **two-phase architecture**: the outer loop tests expensive indexing parameters (chunk_size, chunk_overlap, embedding_model), while the inner loop tests fast query parameters (top_k, temperature). Results are presented with clear rankings showing the best configuration for your data.

The entire process runs locally on the user's machine, with **ChromaDB** storing vectors locally (no Pinecone API key needed). Users can demonstrate the tool by connecting to any database live, proving it works on real data. The final output includes an optimized configuration they can deploy immediately, typically achieving 30-40% cost reduction and 20-35% accuracy improvement over default settings.

## What You're Building

A pip-installable tool that optimizes RAG configurations automatically. Users run it on their machine, connect their database, and get the best RAG setup.

**Not a web platform. A developer tool.**

---

## Core Flow

```bash
# User installs
pip install autorag-optimizer

# User configures (create config.yaml manually)
# See config.yaml.example for template

# User runs optimization
autorag optimize --experiments 20

# User sees results
autorag results --show-report
```

---

## What You'll Build (Essential Only)

### 1. **CLI Interface**

* `autorag optimize` - Run optimization
* `autorag results` - Show results
* `autorag status` - Check progress

**Tech:** Typer library

---

### 2. **Database Connectors**

Support 3 databases:

* Supabase (Storage Bucket)
* MongoDB
* PostgreSQL

**What they do:**

* Validate connection
* Fetch documents
* Handle pagination

---

### 3. **Configuration File**

`config.yaml` user creates:

```yaml
database:
  type: supabase  # Options: supabase, mongodb, postgresql
  url: https://xxx.supabase.co
  key: xxx
  bucket: pdf
  folder: pdf

llm:
  provider: groq  # Options: groq, openai, openrouter
  model: null     # null = use provider default

api_keys:
  groq: sk-xxx       # Required if llm.provider=groq
  openai: sk-xxx     # Required if llm.provider=openai
  openrouter: sk-xxx # Required if llm.provider=openrouter

# RAG Parameter Search Space (NEW!)
rag:
  chunk_size: [256, 500, 1024]        # Characters per chunk
  chunk_overlap: [25, 50, 100]        # Overlap between chunks
  embedding_model:                     # HuggingFace models
    - all-MiniLM-L6-v2
  top_k: [3, 5, 10]                   # Documents to retrieve
  temperature: [0.3, 0.7, 1.0]        # LLM creativity (0-2)

optimization:
  strategy: bayesian  # Options: grid, bayesian
  num_experiments: 20
  test_questions: 50

evaluation:
  method: custom      # Options: custom, ragas
```

---

### 4. **Synthetic Q&A Generator**

* Takes documents
* Generates 50 Q&A pairs
* Validates quality
* Saves to JSON

**Simple, not perfect.**

---

### 5. **Optimization Engine**

**Two-Phase Architecture:**

**Outer Loop** (expensive - requires re-indexing):
* chunk_size variations
* chunk_overlap variations
* embedding_model variations

**Inner Loop** (fast - same index):
* top_k variations
* temperature variations

**Optimizers:**
* **Grid Search** - Tests all combinations systematically
* **Bayesian** (Optuna) - Intelligent sampling with caching

---

### 6. **Vector Store**

**ChromaDB** (Local, No API Key):
* Stores vectors in `.autorag_cache/`
* Auto-detects embedding dimension
* Creates unique collections per config (e.g., `autorag_c500_o50_minilm`)
* No Pinecone API key required

---

### 7. **Evaluation System**

Two evaluation options:

* **Custom** (default): Built-in token-optimized evaluator
* **RAGAS**: Official RAGAS library wrapper (requires `pip install ragas`)

Metrics (both options):
* Answer Relevancy
* Faithfulness  
* Answer Similarity
* Context Recall

---

### 8. **Results Output**

Generate:

* Terminal table (best configs with all 5 parameters)
* JSON file (detailed results)
* HTML report (styled dark theme)

---

## Tech Stack

**Core:**

* Python 3.10+
* Typer (CLI)
* Pydantic (config validation)
* YAML (config files)

**RAG:**

* ChromaDB (local vectors - no API key!)
* sentence-transformers (embeddings)
* Groq / OpenAI / OpenRouter (LLM)

**Optional:**

* Celery + Redis (async background)
* RAGAS (evaluation library)

**Packaging:**

* uv (dependency management)
* pyproject.toml

---

## Project Structure

```
autorag-optimizer/
├── autorag/
│   ├── cli.py              # CLI commands
│   ├── database/
│   │   ├── supabase.py
│   │   ├── mongodb.py
│   │   └── postgres.py
│   ├── rag/  
│   │   ├── embeddings.py     # HuggingFace sentence-transformers
│   │   ├── chroma_store.py   # ChromaDB (replaces vector_store.py)
│   │   ├── llm_client.py     # Multi-provider LLM client
│   │   └── pipeline.py       # RAG pipeline orchestration
│   ├── optimization/
│   │   ├── grid_search.py    # Two-phase grid search
│   │   └── bayesian.py       # Two-phase Bayesian with caching
│   ├── evaluation/
│   │   ├── base_evaluator.py    # Abstract interface
│   │   ├── custom_eval.py       # Built-in evaluator
│   │   ├── ragas_eval.py        # RAGAS wrapper
│   │   └── evaluator_factory.py # Factory function
│   ├── synthetic/
│   │   └── generator.py
│   └── utils/
│       ├── config.py         # Includes RAGConfig model
│       └── text_utils.py     # Chunking logic
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_text_utils.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_pipeline.py
│   ├── test_grid_search.py
│   └── test_bayesian.py
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## Key Decisions

### **What's Essential:**

✅ 3 database connectors

✅ 5 configurable RAG parameters

✅ Two-phase optimization (indexing + query)

✅ Local ChromaDB (no Pinecone)

✅ Synthetic Q&A works

✅ Clear results with all parameters

✅ 45 passing tests

---

## Success Criteria

**Technical:**

* Installs via pip ✓
* Works on 3 databases ✓
* Optimizes 5 RAG parameters ✓
* Uses local ChromaDB (no API key) ✓
* Finds better config than default ✓
* 45 unit tests passing ✓

**Professional:**

* Published on PyPI ✓
* Clear documentation ✓
* Clean error messages ✓
* Can demo live ✓

---

## Resume Line

> "Published AutoRAG Optimizer to PyPI - a CLI tool that automates RAG hyperparameter optimization using Bayesian search across 5 parameters (chunk_size, overlap, embedding_model, top_k, temperature). Uses two-phase architecture with local ChromaDB. Achieves 30-40% cost reduction and 20-35% accuracy improvement. Supports Supabase, MongoDB, PostgreSQL."

---

## What Makes This Good

1. **Actually useful** - Solves real problem
2. **Easy to use** - `pip install` → works
3. **No external dependencies** - ChromaDB runs locally
4. **Comprehensive optimization** - 5 parameters, not just 2
5. **Smart architecture** - Two-phase avoids redundant re-indexing
6. **Production thinking** - Tests, error handling, logging
7. **Publishable** - On PyPI like real packages
