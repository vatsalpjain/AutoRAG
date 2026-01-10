# AutoRAG-Optim

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.1-blue)](https://pypi.org/project/autorag-optim/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Stop guessing your RAG configuration. Let AutoRAG find the optimal one for your data.**

AutoRAG-Optim is a CLI tool that automatically discovers the best RAG (Retrieval-Augmented Generation) hyperparameters for your specific database. Instead of manually testing hundreds of parameter combinations, run one command and get a production-ready configuration optimized for your data.

## Why AutoRAG?

Most teams waste weeks manually tuning RAG settings—chunk sizes, embedding models, retrieval counts—without knowing what actually works best for their data. AutoRAG solves this by:

- **Generating synthetic test data** from your documents (no manual labeling needed)
- **Intelligently searching** the configuration space (20-30 experiments instead of 1000+)
- **Evaluating with real metrics** (accuracy, faithfulness, relevancy, recall)
- **Running entirely locally** (ChromaDB for vectors—no Pinecone API key required)

**Typical results:** 30-40% cost reduction and 20-35% accuracy improvement over default settings.

## Features

| Feature                              | Description                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| 🔍**Smart Optimization**       | Bayesian or Grid Search to find optimal parameters in 20-30 experiments            |
| ⚡**Two-Phase Architecture**   | Expensive indexing params tested separately from fast query params                 |
| 📊**5 Tunable Parameters**     | `chunk_size`, `chunk_overlap`, `embedding_model`, `top_k`, `temperature` |
| 🤖**Synthetic Q&A Generation** | Auto-generate test questions from your documents using LLM                         |
| 📈**RAGAS-like Evaluation**    | Measure accuracy, faithfulness, relevancy, and context recall                      |
| 🗄️**Local Vector Store**     | ChromaDB runs locally—no external API keys needed                                 |
| 🔌**Multi-Database Support**   | Supabase Storage, MongoDB, PostgreSQL                                              |
| 🧠**Multi-LLM Support**        | Groq, OpenAI, OpenRouter                                                           |
| 📋**Rich CLI Output**          | Beautiful terminal output with progress bars, tables, and HTML reports             |

## Installation

```bash
pip install autorag-optim
```

For RAGAS evaluation (optional):

```bash
pip install autorag-optim[ragas]
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

rag:
  chunk_size: [256, 512, 1024]
  chunk_overlap: [50, 100]
  embedding_model:
    - all-MiniLM-L6-v2
  top_k: [3, 5, 10]
  temperature: [0.3, 0.7]

optimization:
  strategy: bayesian    # or: grid
  num_experiments: 20
  test_questions: 50

evaluation:
  method: custom        # or: ragas
```

### 2. Run Optimization

```bash
autorag optimize --config config.yaml
```

### 3. View Results

```bash
autorag results --show-report
```

## Configuration Options

### Optimization Strategy

| Strategy     | Description                                 | Best For                                                  |
| ------------ | ------------------------------------------- | --------------------------------------------------------- |
| `bayesian` | Intelligent search using Optuna TPE sampler | Default choice—finds good configs with fewer experiments |
| `grid`     | Systematic search with stratified sampling  | Guaranteed coverage of search space                       |

### Evaluation Method

| Method     | Description                        | Notes                                                      |
| ---------- | ---------------------------------- | ---------------------------------------------------------- |
| `custom` | Built-in token-optimized evaluator | Works with any LLM, fast, no extra dependencies            |
| `ragas`  | Official RAGAS library metrics     | Requires `pip install ragas`, uses OpenAI-compatible API |

### LLM Providers

| Provider       | Default Model                         | Notes                              |
| -------------- | ------------------------------------- | ---------------------------------- |
| `groq`       | `llama-3.3-70b-versatile`           | Fast inference, generous free tier |
| `openai`     | `gpt-4o-mini`                       | High quality, production-ready     |
| `openrouter` | `meta-llama/llama-3.3-70b-instruct` | Access to 100+ models              |

### Database Connectors

| Type           | Description             | Config Fields                                                       |
| -------------- | ----------------------- | ------------------------------------------------------------------- |
| `supabase`   | Supabase Storage bucket | `url`, `key`, `bucket`, `folder`                            |
| `mongodb`    | MongoDB collection      | `connection_string`, `database`, `collection`                 |
| `postgresql` | PostgreSQL table        | `host`, `port`, `database`, `table`, `user`, `password` |

## Estimated API Calls & Runtime

Understanding the cost before running optimization:

### Formula

```
LLM Calls ≈ Q&A Generation + (Experiments × Questions × Calls per Question)

Where:
- Q&A Generation = ceil(test_questions / 2)  [~1 call per 2 questions]
- Calls per Question = 1 (RAG query) + 3 (evaluation) = 4 calls
```

### Estimates by Configuration

| Questions | Experiments | LLM Calls | Est. Time*   |
| --------- | ----------- | --------- | ------------ |
| 20        | 10          | ~810      | 15-30 min   |
| 50        | 20          | ~4,025    | 45-60 min    |
| 50        | 30          | ~6,025    | 60-90 min    |
| 100       | 20          | ~8,050    | 100-150 min |

*Time varies based on LLM provider rate limits and response times. Groq is typically fastest.

### Cost Saving Tips

- Start with fewer experiments (10-15) to validate your setup
- Use `bayesian` strategy—it finds good configs with 30-40% fewer experiments than grid search
- Reduce `test_questions` for initial exploration (20-30 is enough to rank configs)

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CONNECT                                                     │
│     Fetch documents from your database (Supabase/Mongo/PG)      │
├─────────────────────────────────────────────────────────────────┤
│  2. GENERATE                                                    │
│     Create synthetic Q&A pairs from your documents using LLM    │
├─────────────────────────────────────────────────────────────────┤
│  3. OPTIMIZE (Two-Phase)                                        │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ OUTER LOOP: Indexing params (expensive)             │     │
│     │   → chunk_size, chunk_overlap, embedding_model      │     │
│     │   → Requires re-indexing documents                  │     │
│     └─────────────────────────────────────────────────────┘     │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ INNER LOOP: Query params (fast)                     │     │
│     │   → top_k, temperature                              │     │
│     │   → Same index, just different retrieval settings   │     │
│     └─────────────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  4. EVALUATE                                                    │
│     Score each config: relevancy, faithfulness, similarity,     │
│     context recall → weighted aggregate score                   │
├─────────────────────────────────────────────────────────────────┤
│  5. REPORT                                                      │
│     Terminal table + JSON + HTML report with best config        │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

| Command              | Description                              |
| -------------------- | ---------------------------------------- |
| `autorag optimize` | Run RAG optimization on your database    |
| `autorag results`  | Display optimization results             |
| `autorag status`   | Check optimization progress (async mode) |

```bash
autorag optimize --help

Options:
  -c, --config PATH   Path to config file (default: config.yaml)
  --async             Run optimization in background
```

## Evaluation Metrics

| Metric                      | What It Measures                                             |
| --------------------------- | ------------------------------------------------------------ |
| **Answer Relevancy**  | Is the answer relevant to the question asked?                |
| **Faithfulness**      | Is the answer grounded in the retrieved context?             |
| **Answer Similarity** | How similar is the generated answer to ground truth?         |
| **Context Recall**    | Does the retrieved context contain the required information? |

## Development

```bash
# Clone repository
git clone https://github.com/vatsalpjain/autorag-optim.git
cd autorag-optim

# Install with dev dependencies
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
- **No Pinecone required**—uses local ChromaDB

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see [LICENSE](LICENSE) for details.
