# Curio Agent SDK

A composable, async-first, provider-native agent harness for building production-grade AI agents.

## Overview

Curio Agent SDK provides everything you need to build autonomous agents:

- **Composable agent loops** - ToolCallingLoop (default) or build your own
- **Async-first** - Full async/await with sync wrappers for convenience
- **Native tool calling** - Provider-native function calling (OpenAI, Anthropic, Groq, Ollama)
- **Message-based LLM interface** - Proper system/user/assistant/tool message roles
- **Tiered model routing** - Automatic selection with failover across providers
- **Middleware pipeline** - Logging, cost tracking, tracing, guardrails, rate limiting
- **Memory system** - Conversation, vector, key-value, and composite memory
- **Built-in tools** - Web fetch, file I/O, code execution, HTTP requests
- **Testing utilities** — MockLLM, AgentTestHarness, ToolTestKit, pytest fixtures, record/replay, snapshot testing, BenchmarkSuite, AgentCoverageTracker (tool/hook/error-path coverage), and GitHub Actions workflow template
- **Checkpointing** - Save and resume agent runs
- **Human-in-the-loop** - Tool confirmation before execution
- **Context management** - Token budget fitting with multiple strategies

## Quick Start

### 5-Line Agent

```python
from curio_agent_sdk import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return "Results for: " + query

agent = Agent(
    model="openai:gpt-4o",
    tools=[search],
    system_prompt="You are a helpful research assistant.",
)

result = agent.run("What are the latest developments in quantum computing?")
print(result.output)
```

### Full Configuration

```python
from curio_agent_sdk import (
    Agent, ToolCallingLoop, LLMClient, TieredRouter,
    CostTracker, LoggingMiddleware, TracingMiddleware, GuardrailsMiddleware,
    ConversationMemory, MemoryManager, FileStateStore, ContextManager, CLIHumanInput,
)

agent = Agent(
    loop=ToolCallingLoop(tier="tier3"),
    llm=LLMClient(router=TieredRouter()),
    tools=[search, calculator, fetch_data],
    system_prompt="You are a research agent.",
    agent_id="research-agent",
    max_iterations=25,
    timeout=300,
    context_manager=ContextManager(max_tokens=128000),
    middleware=[
        LoggingMiddleware(),
        CostTracker(budget=1.00),
        TracingMiddleware(service_name="my-agent"),
        GuardrailsMiddleware(block_patterns=[r"(?i)password"]),
    ],
    memory_manager=MemoryManager(memory=ConversationMemory(max_entries=50)),
    state_store=FileStateStore("./state"),
    human_input=CLIHumanInput(),
)

result = await agent.arun("Research quantum computing advances")
print(f"Status: {result.status}")
print(f"Output: {result.output}")
print(f"Tokens: {result.total_input_tokens + result.total_output_tokens}")
```

## Installation

```bash
pip install -e ./curio_agent_sdk

# Provider-specific
pip install openai anthropic groq

# Optional
pip install opentelemetry-api  # For TracingMiddleware
```

## Configuration

### Environment Variables

```bash
# Provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Tier configuration (optional - auto-detected from available keys)
TIER1_MODELS=groq:llama-3.1-8b-instant,openai:gpt-4o-mini
TIER2_MODELS=openai:gpt-4o,anthropic:claude-sonnet-4-6
TIER3_MODELS=anthropic:claude-sonnet-4-6,openai:gpt-4o

# Database (optional)
DB_TYPE=sqlite
DB_PATH=./agent.db
```

### Model String Format

Use `provider:model` shorthand:

```python
agent = Agent(model="openai:gpt-4o", ...)
agent = Agent(model="anthropic:claude-sonnet-4-6", ...)
agent = Agent(model="groq:llama-3.1-70b-versatile", ...)
agent = Agent(model="ollama:llama3.1:8b", ...)
```

## Architecture

```
curio_agent_sdk/
├── __init__.py                     # Public API (v0.6.0)
├── exceptions.py                   # Custom exception hierarchy
├── core/
│   ├── agent.py                    # Main Agent class
│   ├── state.py                    # AgentState
│   ├── context.py                  # ContextManager (token budgets)
│   ├── checkpoint.py               # Checkpoint serialization
│   ├── state_store.py              # StateStore & persistence
│   ├── human_input.py              # Human-in-the-loop
│   ├── loops/
│   │   ├── base.py                 # AgentLoop ABC
│   │   └── tool_calling.py         # Standard tool calling loop
│   └── tools/
│       ├── tool.py                 # Tool class & @tool decorator
│       ├── schema.py               # ToolSchema (JSON Schema generation)
│       ├── registry.py             # ToolRegistry
│       └── executor.py             # Async ToolExecutor
├── llm/
│   ├── client.py                   # LLMClient (async, message-based)
│   ├── router.py                   # TieredRouter with failover
│   ├── token_counter.py            # Token counting
│   └── providers/
│       ├── base.py                 # LLMProvider ABC
│       ├── openai.py               # OpenAI (tools, streaming, vision)
│       ├── anthropic.py            # Anthropic (tools, streaming)
│       ├── groq.py                 # Groq
│       └── ollama.py               # Ollama (on-premise)
├── models/
│   ├── llm.py                      # Message, ToolCall, LLMRequest/Response
│   ├── agent.py                    # AgentRun, AgentRunResult
│   └── events.py                   # EventType, StreamEvent, AgentEvent
├── middleware/
│   ├── base.py                     # Middleware ABC & MiddlewarePipeline
│   ├── logging_mw.py               # Structured logging
│   ├── cost_tracker.py             # Cost tracking & budgets
│   ├── rate_limit.py               # Rate limit handling
│   ├── tracing.py                  # OpenTelemetry tracing & metrics
│   └── guardrails.py               # Content safety filtering
├── memory/
│   ├── base.py                     # Memory ABC
│   ├── conversation.py             # Sliding window memory
│   ├── vector.py                   # Semantic search (embeddings)
│   ├── key_value.py                # Key-value store
│   └── composite.py                # Combine multiple memory types
├── persistence/
│   ├── base.py                     # BasePersistence ABC
│   ├── sqlite.py                   # SQLite backend
│   ├── postgres.py                 # PostgreSQL backend
│   └── memory.py                   # In-memory backend
├── tools/                          # Built-in tools
│   ├── web.py                      # web_fetch
│   ├── code.py                     # python_execute, shell_execute
│   ├── file.py                     # file_read, file_write
│   └── http.py                     # http_request
├── testing/                        # Testing utilities
│   ├── mock_llm.py                 # MockLLM, text_response, tool_call_response
│   ├── harness.py                  # AgentTestHarness
│   ├── fixtures.py                 # Pytest fixtures (mock_llm, agent_test_harness, in-memory stores)
│   ├── coverage.py                 # AgentCoverageTracker (tool/hook/error-path coverage)
│   ├── replay.py                   # RecordingMiddleware, ReplayLLMClient
│   ├── toolkit.py                  # ToolTestKit
│   ├── integration.py             # MultiAgentTestHarness
│   ├── snapshot.py                # SnapshotTester
│   ├── benchmark.py               # BenchmarkSuite
│   ├── eval.py                    # AgentEvalSuite
│   └── regression.py              # RegressionDetector
└── config/                         # Minimal stub (see roadmap)
```

## Core Concepts

### Agent Loops

The SDK provides composable loop patterns:

```python
# Standard tool calling (default)
from curio_agent_sdk import ToolCallingLoop
agent = Agent(loop=ToolCallingLoop(), ...)

# Custom loop
from curio_agent_sdk import AgentLoop, AgentState

class MyLoop(AgentLoop):
    async def step(self, state: AgentState) -> AgentState:
        # Your custom logic
        return state

    def should_continue(self, state: AgentState) -> bool:
        return state.iteration < state.max_iterations
```

### Tools

Define tools with the `@tool` decorator:

```python
from curio_agent_sdk import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool(timeout=30, retries=2, require_confirmation=True)
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.text
```

### Built-in Tools

```python
from curio_agent_sdk.tools import web_fetch, file_read, file_write, http_request, python_execute

agent = Agent(
    tools=[web_fetch, file_read, http_request],
    ...
)
```

### Middleware

```python
from curio_agent_sdk import (
    LoggingMiddleware, CostTracker,
    TracingMiddleware, GuardrailsMiddleware,
)

agent = Agent(
    middleware=[
        LoggingMiddleware(),
        CostTracker(budget=1.00),
        TracingMiddleware(service_name="my-agent"),
        GuardrailsMiddleware(
            block_patterns=[r"(?i)password", r"(?i)secret"],
            block_input_patterns=[r"(?i)ignore previous"],
        ),
    ],
    ...
)
```

### Memory

```python
from curio_agent_sdk import ConversationMemory, CompositeMemory, KeyValueMemory, MemoryManager

memory = CompositeMemory({
    "conversation": ConversationMemory(max_entries=50),
    "knowledge": KeyValueMemory(),
})

agent = Agent(
    memory_manager=MemoryManager(memory=memory),
    ...
)
```

### Streaming

```python
async for event in agent.astream("Tell me about quantum computing"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call_start":
        print(f"\n[Calling {event.tool_name}...]")
    elif event.type == "done":
        print("\n[Done]")
```

### Testing

Use **MockLLM** and **AgentTestHarness** for deterministic tests. For pytest, register the SDK fixtures in your `conftest.py` and use **AgentCoverageTracker** for tool/hook/error-path coverage reporting. A **GitHub Actions** workflow template is provided in `.github/workflows/agent-tests.yml`.

```python
from curio_agent_sdk.testing import MockLLM, AgentTestHarness, tool_call_response

mock = MockLLM()
mock.add_response(tool_call_response("calculate", {"expression": "2+2"}))
mock.add_text_response("2 + 2 = 4")

harness = AgentTestHarness(agent, llm=mock)
result = harness.run_sync("What is 2+2?")

assert result.status == "completed"
assert "4" in result.output
assert harness.tool_calls == [("calculate", {"expression": "2+2"})]
assert harness.llm_calls == 2
```

**Pytest fixtures:** Add `pytest_plugins = ["curio_agent_sdk.testing.fixtures"]` to your `conftest.py` to use `mock_llm`, `agent_test_harness`, `in_memory_state_store`, `in_memory_persistence`, and `tool_test_kit` fixtures.

**Coverage reporting:** Register `AgentCoverageTracker` with your agent's `HookRegistry` to record tools called, hooks emitted, and error paths hit; call `get_report()` or `print_report()` after runs.

### Checkpointing & Resume

```python
from curio_agent_sdk import FileStateStore

agent = Agent(
    state_store=FileStateStore("./state"),
    checkpoint_interval=1,
    ...
)

# Run (saves state)
result = await agent.arun("Long task...")

# Resume from saved state
result = await agent.arun("Continue...", resume_from=result.run_id)
```

## Tier System

Three tiers for different task complexities:

| Tier | Purpose | Default Models |
|------|---------|----------------|
| tier1 | Fast, simple tasks | gpt-4o-mini, llama-3.1-8b |
| tier2 | Balanced quality/speed | gpt-4o, claude-sonnet-4-6 |
| tier3 | High-quality output | gpt-4o, claude-sonnet-4-6 |

## Observability

The SDK provides comprehensive observability through:

- **Event callbacks** - `on_event` parameter for real-time monitoring
- **Database persistence** - SQLite/PostgreSQL for all runs, events, LLM calls
- **OpenTelemetry** - Distributed tracing and metrics via TracingMiddleware
- **Cost tracking** - Per-model pricing with budget enforcement
- **Structured logging** - LoggingMiddleware for all operations

A separate **[Curio Agent Observability](../curio_agent_observability/)** dashboard provides a web UI for exploring agent activity.

## License

Apache License 2.0
