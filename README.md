# AutoRAG-Optim

**Automatically find the optimal RAG configuration for your database.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoRAG-Optim is a CLI tool that automates RAG (Retrieval-Augmented Generation) hyperparameter optimization. Connect your database, run optimization, and get the best RAG configuration in 4-6 hours.

## Features

- 🔍 **Automated Optimization** - Bayesian or Grid Search to find optimal RAG parameters
- 📊 **Synthetic Q&A Generation** - Auto-generate test questions from your documents
- 📈 **RAGAS-like Metrics** - Evaluate accuracy, faithfulness, relevancy, and context recall
- 🗄️ **Multi-Database Support** - Supabase, MongoDB, PostgreSQL
- 🤖 **Multi-LLM Support** - Groq, OpenAI, OpenRouter
- 📋 **Rich CLI Output** - Beautiful terminal output with progress bars and tables

## Installation

```bash
pip install autorag-optim
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
  pinecone: your-pinecone-api-key
  pinecone_index: autorag

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
2. **Generate** - Creates synthetic Q&A pairs using LLM
3. **Optimize** - Tests multiple RAG configurations (top_k, temperature)
4. **Evaluate** - Measures accuracy, cost, and latency for each config
5. **Report** - Shows best configurations with Pareto frontier

## Metrics

| Metric | Description |
|--------|-------------|
| Answer Relevancy | How relevant is the answer to the question? |
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Similarity | How similar is the answer to ground truth? |
| Context Recall | Does the context contain the required info? |
| Cost | Token usage estimation |
| Latency | Response time |

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/autorag-optim.git
cd autorag-optim

# Install with uv
uv sync

# Run CLI
uv run autorag --help
```

## Requirements

- Python 3.10+
- Pinecone account (vector store)
- LLM API key (Groq, OpenAI, or OpenRouter)
- Database (Supabase, MongoDB, or PostgreSQL)

## License

MIT License - see [LICENSE](LICENSE) for details.
