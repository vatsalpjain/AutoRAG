# AutoRAG-Optim

**Automatically find the optimal RAG configuration for your database.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoRAG-Optim is a CLI tool that automates RAG (Retrieval-Augmented Generation) hyperparameter optimization. Connect your database, run optimization, and get the best RAG configuration in minutes to hours.

## Features

- 🔍 **Automated Optimization** - Bayesian or Grid Search to find optimal RAG parameters
- 📊 **5 Configurable Parameters** - chunk_size, chunk_overlap, embedding_model, top_k, temperature
- 🗄️ **Local Vector Store** - ChromaDB (no API key needed, runs locally)
- 📝 **Synthetic Q&A Generation** - Auto-generate test questions from your documents
- 📈 **RAGAS-like Metrics** - Evaluate accuracy, faithfulness, relevancy, and context recall
- 🗄️ **Multi-Database Support** - Supabase, MongoDB, PostgreSQL
- 🤖 **Multi-LLM Support** - Groq, OpenAI, OpenRouter
- 📋 **Rich CLI Output** - Beautiful terminal output with progress bars and tables

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autorag-optim.git
cd autorag-optim

# Install with uv
uv sync
```

## Quick Start

### 1. Create Configuration

Create a `config.yaml` file:

```yaml
database:
  type: supabase
  url: https://your-project.supabase.co
  key: your-supabase-anon-key
  bucket: pdf
  folder: pdf

llm:
  provider: groq
  model: null  # Uses default: llama-3.3-70b-versatile

api_keys:
  groq: your-groq-api-key

# RAG Parameter Search Space (all parameters are lists to search over)
rag:
  chunk_size: [256, 500, 1024]
  chunk_overlap: [25, 50, 100]
  embedding_model:
    - all-MiniLM-L6-v2
  top_k: [3, 5, 10]
  temperature: [0.3, 0.7, 1.0]

optimization:
  strategy: bayesian
  num_experiments: 20
  test_questions: 50

evaluation:
  method: custom
```

### 2. Run Optimization

```bash
autorag optimize --config config.yaml
```

### 3. View Results

```bash
autorag results --show-report
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `autorag optimize` | Run RAG optimization on your database |
| `autorag results` | Display optimization results |
| `autorag status` | Check optimization progress |

### Options

```bash
autorag optimize --help

Options:
  -c, --config PATH   Path to config file (default: config.yaml)
  --async             Run optimization in background
```

## Configuration

### RAG Parameters

AutoRAG optimizes 5 RAG parameters in a **two-phase architecture**:

**Indexing Parameters** (require re-indexing, tested in outer loop):
```yaml
rag:
  chunk_size: [256, 500, 1024]     # Characters per chunk
  chunk_overlap: [25, 50, 100]     # Overlap between chunks
  embedding_model:                  # HuggingFace model names
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
```

**Query Parameters** (fast, tested in inner loop):
```yaml
rag:
  top_k: [3, 5, 10]                # Documents to retrieve
  temperature: [0.3, 0.7, 1.0]     # LLM creativity (0-2)
```

### Database Options

**Supabase (Storage Bucket)**
```yaml
database:
  type: supabase
  url: https://xxx.supabase.co
  key: your-key
  bucket: pdf
  folder: pdf
```

**MongoDB**
```yaml
database:
  type: mongodb
  connection_string: mongodb://localhost:27017
  database: your_db
  collection: documents
```

**PostgreSQL**
```yaml
database:
  type: postgresql
  host: localhost
  port: 5432
  database: your_db
  table: documents
  user: username
  password: password
```

### LLM Providers

```yaml
llm:
  provider: groq      # groq | openai | openrouter
  model: null         # null = use provider default

api_keys:
  groq: sk-xxx        # Required if provider=groq
  openai: sk-xxx      # Required if provider=openai
  openrouter: sk-xxx  # Required if provider=openrouter
```

### Evaluation Methods

```yaml
evaluation:
  method: custom   # custom | ragas
```

- **custom** (default): Built-in token-optimized evaluator
- **ragas**: Official RAGAS library (requires `pip install ragas`)

## How It Works

1. **Connect** - Fetches documents from your database
2. **Display Search Space** - Shows all RAG parameter combinations to test
3. **Generate** - Creates synthetic Q&A pairs using LLM
4. **Optimize** - Two-phase optimization:
   - Outer loop: Tests indexing parameters (chunk_size, overlap, embedding_model)
   - Inner loop: Tests query parameters (top_k, temperature) on each index
5. **Evaluate** - Measures accuracy using RAGAS-like metrics
6. **Report** - Shows best configuration with all parameters

## Vector Store

AutoRAG uses **ChromaDB** for local vector storage:

- ✅ **No API key required** - Runs entirely locally
- ✅ **Automatic dimension detection** - Works with any embedding model
- ✅ **Persistent storage** - Vectors saved in `.autorag_cache/`
- ✅ **Dynamic collections** - Separate index for each config (e.g., `autorag_c500_o50_minilm`)

## Metrics

| Metric | Description |
|--------|-------------|
| Answer Relevancy | How relevant is the answer to the question? |
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Similarity | How similar is the answer to ground truth? |
| Context Recall | Does the context contain the required info? |

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/autorag-optim.git
cd autorag-optim

# Install with uv
uv sync --extra dev

# Run CLI
uv run autorag --help

# Run tests
uv run pytest tests/ -v
```

## Requirements

- Python 3.10+
- LLM API key (Groq, OpenAI, or OpenRouter)
- Database (Supabase, MongoDB, or PostgreSQL)
- **No Pinecone required** - Uses local ChromaDB

## License

MIT License - see [LICENSE](LICENSE) for details.
