# Environment Configuration Examples

This directory contains example `.env` files for different setup scenarios.

## Quick Start

1. **Recommended**: Start with `examples/recommended_template.env` for a production-ready setup
2. Copy the example that matches your needs to your project root as `.env`
3. Replace the placeholder API keys with your actual keys
4. Run your agent!

## Available Examples

### `recommended_template.env` ⭐ **RECOMMENDED**
**Location:** `examples/recommended_template.env` (in the examples folder, not env_examples)
**Best for:** Production-ready setup with best practices

Comprehensive template with:
- SQLite database (with PostgreSQL option)
- Multi-provider configuration (Groq, OpenAI, Anthropic)
- Optimized tier routing for all three tiers
- Multi-key rotation examples
- All configuration options documented

This is the recommended starting point for most use cases.

### `minimal_auto_detect.env`
**Best for:** Quickest start, just testing

Just set one API key - the SDK auto-configures sensible defaults:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### `simple_single_provider.env`
**Best for:** Getting started with explicit control

One provider with explicit tier models:

```bash
OPENAI_API_KEY=sk-your-key-here

TIER1_MODELS=openai:gpt-4o-mini
TIER2_MODELS=openai:gpt-4o
TIER3_MODELS=openai:gpt-4o
```

### `single_model_all_tiers.env`
**Best for:** Consistent behavior, testing

Same model for everything:

```bash
OPENAI_API_KEY=sk-your-key-here

TIER1_MODELS=openai:gpt-4o
TIER2_MODELS=openai:gpt-4o
TIER3_MODELS=openai:gpt-4o
```

### `single_provider_tiered.env`
**Best for:** Cost optimization with one provider

Different models for different task complexities:

```bash
OPENAI_API_KEY=sk-your-key-here

TIER1_MODELS=openai:gpt-4o-mini      # Fast, cheap
TIER2_MODELS=openai:gpt-4o           # Balanced
TIER3_MODELS=openai:gpt-4o           # Best quality
```

### `multi_provider_failover.env`
**Best for:** Production, high reliability

Multiple providers with automatic failover. Order IS priority:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# If Groq fails, try OpenAI
TIER1_MODELS=groq:llama-3.1-8b-instant,openai:gpt-4o-mini

# Mix providers in any order
TIER2_MODELS=groq:llama-3.3-70b-versatile,openai:gpt-4o,anthropic:claude-3-5-sonnet-20241022
```

### `local_ollama.env`
**Best for:** Privacy, offline use, no API costs

Run models locally with Ollama:

```bash
OLLAMA_HOST=http://localhost:11434

TIER1_MODELS=ollama:llama3.1:8b
TIER2_MODELS=ollama:llama3.1:70b
TIER3_MODELS=ollama:llama3.1:70b
```

## Configuration Reference

### API Keys

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Ollama | `OLLAMA_HOST` (no key needed) |

### Tier Configuration

The model list order IS the priority. First model is tried first; if it fails, next is tried.

```bash
# Format: provider:model,provider:model,provider:model
TIER1_MODELS=openai:gpt-4o-mini,groq:llama-3.1-8b-instant
TIER2_MODELS=openai:gpt-4o
TIER3_MODELS=anthropic:claude-3-5-sonnet-20241022,openai:gpt-4o
```

### Auto-Detection

If no `TIER*_MODELS` are set, the SDK auto-configures based on available API keys:

| Provider | tier1 (fast) | tier2 (balanced) | tier3 (best) |
|----------|--------------|------------------|--------------|
| OpenAI | gpt-4o-mini | gpt-4o | gpt-4o |
| Anthropic | claude-3-haiku | claude-3.5-sonnet | claude-3.5-sonnet |
| Groq | llama-3.1-8b-instant | llama-3.3-70b | llama-3.3-70b |
| Ollama | llama3.1:8b | llama3.1:70b | llama3.1:70b |

### Multi-Key Rotation

For high-volume usage, configure multiple API keys:

```bash
GROQ_API_KEY_1=gsk_first-key
GROQ_API_KEY_1_NAME=primary
GROQ_API_KEY_2=gsk_second-key
GROQ_API_KEY_2_NAME=secondary
```

### Database Configuration

```bash
# SQLite (default, simplest)
DB_TYPE=sqlite
DB_PATH=./agent.db

# PostgreSQL (production)
DB_TYPE=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agent_db
DB_USER=postgres
DB_PASSWORD=secret

# In-memory (testing)
DB_TYPE=memory
```
