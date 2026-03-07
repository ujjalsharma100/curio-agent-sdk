# Curio Agent SDK — Python examples

These examples show how to run a **tool-calling agent** with different LLM providers. Each run writes a detailed **run log** (`.log` file) so you can manually inspect prompts, tool calls, and results.

## Prerequisites

- Python 3.11+

## Using a virtual environment

From the **repository root** (`curio-agent-sdk-python`):

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install SDK with provider extras (pick what you need)
pip install -e ".[openai,groq,anthropic,ollama]"
# Or per-provider: .[openai], .[groq], .[anthropic], .[ollama]

# Run an example (set the right API key first)
export OPENAI_API_KEY=sk-...
python examples/tool_agent_openai.py
```

Without activating, you can use the venv’s Python directly:

```bash
.venv/bin/python examples/tool_agent_openai.py
```

## Tool agents by provider

| Example | Provider | Model (example) | API key / setup |
|--------|----------|------------------|-----------------|
| `tool_agent_openai.py` | OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `tool_agent_groq.py` | Groq | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `tool_agent_anthropic.py` | Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| `tool_agent_ollama.py` | Ollama | `llama3.1` | Ollama running; optional `OLLAMA_HOST` |

Each example:

- Registers a **run logger** that appends to a timestamped `.log` file in the `examples/` directory.
- Uses shared **calculator** and **search** tools (see `tools.py`).
- Prints the agent output and the path to the log file.

## How to run and validate

Run from the **repository root** so that the `curio_agent_sdk` package is found (editable install) and the examples directory is on the path when executing the script.

### OpenAI

```bash
cd curio-agent-sdk-python
export OPENAI_API_KEY=sk-...
pip install -e ".[openai]"
python examples/tool_agent_openai.py
```

### Groq

```bash
export GROQ_API_KEY=gsk_...
pip install -e ".[groq]"
python examples/tool_agent_groq.py
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install -e ".[anthropic]"
python examples/tool_agent_anthropic.py
```

### Ollama

```bash
# Start Ollama and pull a model that supports tools, e.g. llama3.1
ollama run llama3.1

pip install -e ".[ollama]"
python examples/tool_agent_ollama.py

# Optional: custom host
export OLLAMA_HOST=http://127.0.0.1:11434
python examples/tool_agent_ollama.py
```

## Run log contents

After each run, a file like `tool-agent-openai-2026-03-07T12-00-00.log` is created in the `examples/` directory. It contains:

- **AGENT RUN START** — input, run_id, agent_id
- **LLM REQUEST** — messages and tools sent to the provider (when middleware is used so the pipeline emits these hooks)
- **LLM RESPONSE** — model, content, tool_calls, usage
- **TOOL CALL START / END** — tool name, arguments, result (or error)
- **AGENT RUN END** — output and summary

Use this log to confirm that tool schemas are sent correctly (no `$ref`/`definitions` in parameters), that the LLM receives and returns tool calls, and that the executor runs tools with the right arguments.

## Schema behavior (OpenAI/Groq compatibility)

The Python SDK builds tool **parameters** as a **direct JSON object** (e.g. `type`, `properties`, `required`). It does **not** emit top-level `$ref` or `definitions`, which some providers (e.g. OpenAI) reject. The regression test `test_to_llm_schema_direct_object_no_ref` in `tests/unit/tools/test_tool_schema.py` ensures this stays true.

## Files

- `tools.py` — Shared `calculator` and `search` tools used by all tool-agent examples.
- `tool_agent_openai.py`, `tool_agent_groq.py`, `tool_agent_anthropic.py`, `tool_agent_ollama.py` — Provider-specific entry points.

The **run logger** is an optional utility provided by the SDK. Import it with:

```python
from curio_agent_sdk import use_run_logger, create_run_logger, RunLogger
```
