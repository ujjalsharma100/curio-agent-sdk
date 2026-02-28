# Curio Agent SDK

A composable, async-first, production-grade agent harness for building AI agents — from simple tool-calling bots to coding agents, deep research systems, computer-use agents, and multi-agent orchestrations.

**v0.6.0** — Every component is independently usable and replaceable. Convention over configuration. Zero mandatory dependencies beyond Python stdlib + an LLM provider.

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

### Builder Pattern

```python
from curio_agent_sdk import Agent, SubagentConfig

agent = Agent.builder() \
    .model("anthropic:claude-sonnet-4-6") \
    .system_prompt("You are a research agent.") \
    .tools([search, calculator]) \
    .instructions("Always cite sources.") \
    .instructions_file("./AGENT.md") \
    .hook("tool.call.before", lambda ctx: print(f"Calling {ctx.data['tool']}")) \
    .subagent("researcher", SubagentConfig(
        system_prompt="Research specialist",
        tools=[web_search],
    )) \
    .permissions(AllowReadsAskWrites()) \
    .build()
```

### Full Configuration

```python
from curio_agent_sdk import (
    Agent, ToolCallingLoop, LLMClient, TieredRouter,
    CostTracker, GuardrailsMiddleware, MemoryManager,
    ConversationMemory, FileStateStore, ContextManager,
    CLIHumanInput, SessionManager, InMemorySessionStore,
    TaskManager, InstructionLoader, AllowReadsAskWrites,
    HookRegistry, InMemoryEventBus,
)

agent = Agent(
    loop=ToolCallingLoop(tier="tier3"),
    llm=LLMClient(router=TieredRouter(), dedup_enabled=True),
    tools=[search, calculator, fetch_data],
    system_prompt="You are a research agent.",
    agent_id="research-agent",
    max_iterations=25,
    timeout=300,
    context_manager=ContextManager(max_tokens=128000),
    middleware=[
        CostTracker(budget=1.00, alert_thresholds=[0.5, 0.8]),
        GuardrailsMiddleware(
            block_patterns=[r"(?i)password"],
            block_prompt_injection=True,
        ),
    ],
    memory_manager=MemoryManager(memory=ConversationMemory(max_entries=50)),
    state_store=FileStateStore("./state"),
    session_manager=SessionManager(InMemorySessionStore()),
    human_input=CLIHumanInput(),
    instruction_loader=InstructionLoader(),
    permission_policy=AllowReadsAskWrites(),
    event_bus=InMemoryEventBus(),
)

async with agent:
    result = await agent.arun("Research quantum computing advances")
    print(f"Output: {result.output}")
```

## Installation

From the repo root (or package directory):

```bash
pip install -e .

# Provider-specific
pip install openai anthropic groq

# Optional extras
pip install pydantic>=2.0         # Structured output
pip install pyautogui             # Computer use tools
pip install playwright            # Browser automation
pip install opentelemetry-api     # OpenTelemetry tracing
```

## Configuration

### Environment Variables

```bash
# Provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Tier configuration (optional — auto-detected from available keys)
TIER1_MODELS=groq:llama-3.1-8b-instant,openai:gpt-4o-mini
TIER2_MODELS=openai:gpt-4o,anthropic:claude-sonnet-4-6
TIER3_MODELS=anthropic:claude-sonnet-4-6,openai:gpt-4o
```

### Model String Format

```python
agent = Agent(model="openai:gpt-4o", ...)
agent = Agent(model="anthropic:claude-sonnet-4-6", ...)
agent = Agent(model="groq:llama-3.1-70b-versatile", ...)
agent = Agent(model="ollama:llama3.1:8b", ...)
```

## Architecture

Package uses a **src layout**. Install with `pip install -e .` from repo root.

```
project_root/
├── src/
│   └── curio_agent_sdk/
│       ├── __init__.py                 # Public API (re-exports)
│       ├── base/
│       │   ├── __init__.py
│       │   └── component.py             # Component ABC (lifecycle)
│       ├── credentials/
│       │   ├── __init__.py
│       │   └── credentials.py           # CredentialResolver (Vault, AWS, env)
│       ├── resilience/
│       │   ├── __init__.py
│       │   └── circuit_breaker.py       # CircuitBreaker
│       ├── config/
│       ├── cli/
│       │   ├── __init__.py
│       │   └── cli.py                   # AgentCLI interactive harness
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent/
│       │   │   ├── agent.py             # Agent (thin shell)
│       │   │   ├── builder.py           # AgentBuilder (fluent API)
│       │   │   └── runtime.py           # Runtime (orchestration engine)
│       │   ├── state/
│       │   │   ├── state.py             # AgentState + typed extensions
│       │   │   ├── state_store.py       # StateStore ABC + implementations
│       │   │   ├── checkpoint.py        # Checkpoint serialization
│       │   │   └── session.py           # SessionManager + SessionStore
│       │   ├── context/
│       │   │   ├── context.py           # ContextManager (token budgets)
│       │   │   └── instructions.py      # InstructionLoader (AGENT.md)
│       │   ├── events/
│       │   │   ├── hooks.py             # HookRegistry + HookContext
│       │   │   └── event_bus.py         # EventBus + InMemoryEventBus
│       │   ├── extensions/
│       │   │   ├── plugins.py           # Plugin ABC + discovery
│       │   │   ├── skills.py            # Skill + SkillRegistry
│       │   │   └── subagent.py          # SubagentConfig + AgentOrchestrator
│       │   ├── workflow/
│       │   │   ├── plan_mode.py         # PlanMode + TodoManager
│       │   │   ├── task_manager.py      # TaskManager (long-running tasks)
│       │   │   └── structured_output.py # Pydantic structured output
│       │   ├── security/
│       │   │   ├── permissions.py       # PermissionPolicy + sandbox
│       │   │   └── human_input.py       # Human-in-the-loop
│       │   ├── loops/
│       │   │   ├── base.py              # AgentLoop ABC
│       │   │   └── tool_calling.py      # Standard tool calling loop
│       │   ├── tools/
│       │   │   ├── tool.py              # Tool class + @tool decorator
│       │   │   ├── schema.py            # ToolSchema (JSON Schema)
│       │   │   ├── registry.py          # ToolRegistry
│       │   │   └── executor.py          # Async ToolExecutor
│       │   └── llm/
│       │       ├── client.py            # LLMClient (async, dedup, batch)
│       │       ├── router.py            # TieredRouter + degradation
│       │       ├── token_counter.py    # Token counting (cached)
│       │       ├── batch_client.py      # BatchLLMClient
│       │       └── providers/
│       │           ├── base.py         # LLMProvider ABC
│       │           ├── openai.py        # OpenAI (tools, streaming, vision)
│       │           ├── anthropic.py     # Anthropic (tools, streaming, cache)
│       │           ├── groq.py          # Groq
│       │           └── ollama.py       # Ollama (on-premise)
│       ├── models/
│       │   ├── llm.py                  # Message, ToolCall, LLMRequest/Response
│       │   ├── agent.py                # AgentRun, AgentRunResult
│       │   ├── events.py               # EventType, StreamEvent, AgentEvent
│       │   └── exceptions.py           # Custom exception hierarchy
│       ├── middleware/
│       │   ├── base.py                 # Middleware ABC + MiddlewarePipeline
│       │   ├── logging_mw.py           # Structured logging
│       │   ├── cost_tracker.py         # Cost tracking, budgets, alerts
│       │   ├── rate_limit.py           # Per-user/agent rate limiting
│       │   ├── tracing.py              # OpenTelemetry tracing + metrics
│       │   ├── guardrails.py           # Content safety, PII, injection
│       │   ├── consumers.py            # Hook-based observability consumers
│       │   └── prometheus.py           # Prometheus/Grafana export
│       ├── memory/
│       │   ├── base.py                 # Memory ABC
│       │   ├── manager.py              # MemoryManager + strategies
│       │   ├── conversation.py         # Sliding window memory
│       │   ├── vector.py               # Semantic search (embeddings)
│       │   ├── key_value.py            # Key-value store
│       │   ├── composite.py            # Combine multiple memory types
│       │   ├── working.py              # Ephemeral scratchpad
│       │   ├── episodic.py             # Temporal experience memory
│       │   ├── graph.py                # Entity-relationship knowledge graph
│       │   ├── self_editing.py         # MemGPT/Letta-style core + archival
│       │   ├── file_memory.py          # File-based persistent memory
│       │   └── policies.py             # Decay, importance, summarization
│       ├── persistence/
│       │   ├── base.py                 # BasePersistence ABC + audit logs
│       │   ├── audit_hooks.py           # register_audit_hooks (wire to persistence)
│       │   ├── sqlite.py               # SQLite backend
│       │   ├── postgres.py             # PostgreSQL backend
│       │   └── memory.py               # In-memory backend
│       ├── mcp/
│       │   ├── client.py               # MCPClient (stdio + HTTP)
│       │   ├── transport.py            # StdioTransport, HTTPTransport
│       │   ├── config.py               # MCPServerConfig, load from file
│       │   ├── adapter.py              # MCP → Curio Tool adapter
│       │   └── bridge.py               # MCPBridge (Component lifecycle)
│       ├── connectors/
│       │   ├── base.py                 # Connector ABC + ConnectorResource
│       │   └── bridge.py               # ConnectorBridge (Component lifecycle)
│       ├── tools/                      # Built-in tools
│       │   ├── web.py                  # web_fetch
│       │   ├── code.py                 # python_execute, shell_execute
│       │   ├── file.py                 # file_read, file_write
│       │   ├── http.py                 # http_request
│       │   ├── computer_use.py         # ComputerUseToolkit
│       │   └── browser.py              # BrowserToolkit (Playwright)
│       └── testing/
│           ├── mock_llm.py             # MockLLM, text_response, tool_call_response
│           ├── harness.py              # AgentTestHarness
│           ├── fixtures.py             # Pytest fixtures
│           ├── coverage.py             # AgentCoverageTracker
│           ├── replay.py               # RecordingMiddleware, ReplayLLMClient
│           ├── toolkit.py              # ToolTestKit
│           ├── integration.py          # MultiAgentTestHarness
│           ├── snapshot.py            # SnapshotTester
│           ├── benchmark.py            # BenchmarkSuite
│           ├── eval.py                 # AgentEvalSuite
│           └── regression.py           # RegressionDetector
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Core Concepts

### Agent, Runtime, and Builder

The `Agent` is a thin shell. The `Runtime` handles all orchestration. The `AgentBuilder` provides a fluent API for construction.

```python
from curio_agent_sdk import Agent, Runtime

# Simple constructor
agent = Agent(model="openai:gpt-4o", tools=[search])

# Builder pattern (recommended for complex agents)
agent = Agent.builder() \
    .model("openai:gpt-4o") \
    .tools([search]) \
    .system_prompt("You are helpful.") \
    .build()

# Direct Runtime access for advanced use
state = agent.runtime.create_state("Do something")
result = await agent.runtime.run_with_state(state)
```

### Tools

Define tools with the `@tool` decorator. JSON schemas are auto-generated from type hints.

```python
from curio_agent_sdk import tool, ToolConfig

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool(timeout=30, retries=2, require_confirmation=True, cache_ttl=60)
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        return (await client.get(url)).text

# Idempotent tools skip re-execution on checkpoint restore
@tool(config=ToolConfig(idempotent=True))
async def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    ...
```

### Built-in Tools

```python
from curio_agent_sdk.tools import web_fetch, file_read, file_write, http_request, python_execute
from curio_agent_sdk import ComputerUseToolkit, BrowserToolkit

# Standard tools
agent = Agent(tools=[web_fetch, file_read, http_request], ...)

# Computer use (pip install curio-agent-sdk[computer-use])
agent = Agent(tools=ComputerUseToolkit().get_tools(), ...)

# Browser automation (pip install curio-agent-sdk[browser] && playwright install)
browser = BrowserToolkit(headless=True)
agent = Agent(tools=browser.get_tools(), ...)
```

### Hooks / Lifecycle System

Hooks let you customize agent behavior at every lifecycle point — without subclassing. Hooks are mutable: they can modify context, cancel actions, and inject data.

```python
from curio_agent_sdk import HookRegistry

# Block dangerous tool calls
agent = Agent.builder() \
    .hook("tool.call.before", lambda ctx: ctx.cancel() if ctx.data["tool"] == "rm" else None) \
    .hook("llm.call.after", lambda ctx: log_response(ctx.data["response"])) \
    .hook("agent.run.after", lambda ctx: notify_slack("Agent completed")) \
    .build()

# All hook events:
# agent.run.before/after/error, agent.iteration.before/after,
# llm.call.before/after/error, tool.call.before/after/error,
# memory.inject.before, memory.save.before,
# state.checkpoint.before/after

# Load hooks from config file (YAML/TOML)
# hooks.yaml:
#   - event: tool.call.before
#     handler: my_module:validate_tool_call
#   - event: agent.run.after
#     shell: "echo 'Done' >> /tmp/agent.log"
```

### Rules / Instructions

Hierarchical instruction loading — like `CLAUDE.md` or `.cursorrules`. Global > project > directory.

```python
from curio_agent_sdk import InstructionLoader

# Auto-loads AGENT.md and .agent/rules.md from global, project, and cwd
agent = Agent.builder() \
    .instructions(InstructionLoader()) \
    .build()

# Or manual
agent = Agent.builder() \
    .instructions("Always respond in JSON format.") \
    .instructions_file("./AGENT.md") \
    .build()

# Dynamic injection mid-session
agent.add_instructions("From now on, prefer short answers.")
```

### Skills

Packaged, reusable agent capabilities — bundle prompts, tools, and hooks into named skills.

```python
from curio_agent_sdk import Skill, SkillRegistry

commit_skill = Skill(
    name="commit",
    description="Create well-formatted git commits",
    system_prompt="When committing, analyze changes...",
    tools=[git_status, git_diff, git_add, git_commit],
)

agent = Agent.builder() \
    .skill(commit_skill) \
    .skill(Skill.from_directory("./skills/review-pr")) \
    .build()

# Invoke a skill
result = await agent.invoke_skill("commit", "Commit the auth changes")

# Activate/deactivate skills mid-run
await agent.arun("Plan the feature", active_skills=["planning"])
```

### Subagents / Multi-Agent Orchestration

Spawn specialized subagents, run them in the background, or hand off conversations.

```python
from curio_agent_sdk import Agent, SubagentConfig

agent = Agent.builder() \
    .model("anthropic:claude-sonnet-4-6") \
    .subagent("researcher", SubagentConfig(
        system_prompt="Research specialist",
        tools=[web_search, fetch_page],
        model="openai:gpt-4o",
    )) \
    .subagent("coder", SubagentConfig(
        system_prompt="Expert programmer",
        tools=[read_file, edit_file, run_tests],
        inherit_memory=True,
    )) \
    .build()

# Subagents are available as tools — the parent agent can spawn them
# Or programmatically:
result = await agent.spawn_subagent("researcher", "Find papers on transformers")
task_id = await agent.spawn_subagent_background("coder", "Implement the feature")
result = await agent.get_subagent_result(task_id)

# Handoff conversation to another agent
await agent.handoff(other_agent, "Continue this analysis")
```

### Plan Mode & Todos

Plan-then-execute workflows with task tracking and approval gates.

```python
from curio_agent_sdk import Agent

agent = Agent.builder() \
    .model("openai:gpt-4o") \
    .tools([read_file, edit_file, run_tests]) \
    .plan_mode(read_only_tool_names=["read_file"]) \
    .build()

# Agent can enter plan mode (restricts to read-only tools),
# design a plan, exit with plan for approval, then execute.
# Todos are tracked as part of agent state and persisted in checkpoints.

# Check plan status
if agent.is_awaiting_plan_approval():
    plan = agent.get_plan()
    print(plan)
```

### Structured Output

Get validated Pydantic models back from agent runs.

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    summary: str

result = await agent.arun(
    "Find 3 papers on transformers",
    response_format=list[SearchResult],
)

for paper in result.parsed_output:
    print(f"{paper.title}: {paper.url}")
```

### Memory System

7 memory types with pluggable strategies for injection, saving, and querying.

```python
from curio_agent_sdk import (
    MemoryManager, ConversationMemory, CompositeMemory,
    VectorMemory, KeyValueMemory, WorkingMemory,
    EpisodicMemory, GraphMemory, SelfEditingMemory, FileMemory,
)

# Simple conversation memory
agent = Agent(memory_manager=MemoryManager(memory=ConversationMemory(max_entries=50)), ...)

# Composite memory with multiple backends
memory = CompositeMemory({
    "conversation": ConversationMemory(max_entries=50),
    "knowledge": KeyValueMemory(),
    "semantic": VectorMemory(persist_path="./vectors"),
})

# MemGPT/Letta-style self-editing memory (agent manages its own memory via tools)
memory = SelfEditingMemory()
agent = Agent(
    memory_manager=MemoryManager(memory=memory),
    tools=memory.get_tools(),  # core_memory_read/write, archival_search/insert
    ...
)

# File-based persistent memory (Claude Code style)
memory = FileMemory(base_path="./memory", namespace="project-x")

# Pluggable strategies
from curio_agent_sdk.memory.manager import (
    AdaptiveTokenQuery, SaveSummaryStrategy, UserMessageInjection,
)
manager = MemoryManager(
    memory=memory,
    injection_strategy=UserMessageInjection(),
    save_strategy=SaveSummaryStrategy(summarize_fn=my_summarizer),
    query_strategy=AdaptiveTokenQuery(),
)
```

### Sessions / Conversations

Persistent multi-turn conversations with session management.

```python
from curio_agent_sdk import SessionManager, InMemorySessionStore

session_mgr = SessionManager(InMemorySessionStore())
agent = Agent.builder().session_manager(session_mgr).model("openai:gpt-4o").build()

session = await session_mgr.create(agent.agent_id)
result = await agent.arun("Hello!", session_id=session.id)
# Messages are automatically persisted
result = await agent.arun("Follow up question", session_id=session.id)
# Agent has full conversation history
```

### MCP (Model Context Protocol)

Connect to MCP servers for dynamic tool discovery — supports stdio and HTTP transports, Cursor/Claude-style config, and env var credential resolution.

```python
# From URL
agent = Agent.builder() \
    .mcp_server("stdio://npx -y @modelcontextprotocol/server-filesystem /path") \
    .mcp_server("http://localhost:8080") \
    .build()

# From config (Cursor/Claude style) with credential resolution
agent = Agent.builder() \
    .mcp_server_config({
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_TOKEN"},
    }) \
    .build()

# From config file (mcpServers format)
agent = Agent.builder() \
    .mcp_servers_from_file("mcp.json") \
    .build()

# MCP tools are discovered at startup
async with agent:
    result = await agent.arun("List my GitHub repos")
```

### Connectors

Pluggable connector framework for external services. Implement the `Connector` ABC and register tools are auto-discovered at startup.

```python
from curio_agent_sdk import Connector

class MyAPIConnector(Connector):
    name = "my-api"

    async def connect(self, credentials=None): ...
    async def disconnect(self): ...
    def get_tools(self): return [self.query_api, self.submit_data]
    ...

agent = Agent.builder() \
    .connector(MyAPIConnector(api_key="$MY_API_KEY")) \
    .build()
```

### Long-Running Tasks

Background execution, pause/resume, progress tracking, and concurrency limits.

```python
from curio_agent_sdk import TaskManager

task_mgr = TaskManager(max_concurrent=2)

# Submit background task
task_id = await task_mgr.submit(agent, "Comprehensive analysis of X")

# Track progress
task_mgr.on_progress(task_id, lambda rid, i, max_i: print(f"{i}/{max_i}"))

# Pause and resume
await task_mgr.pause(task_id)
await task_mgr.resume(task_id)

# Wait for completion
result = await task_mgr.wait(task_id, timeout=600)

# Crash recovery — find and resume interrupted runs
recovered = await task_mgr.recover_incomplete(agent, "Continue task")
```

### Permissions & Sandboxing

Composable permission policies with file system and network sandboxing.

```python
from curio_agent_sdk import (
    AllowAll, AskAlways, AllowReadsAskWrites,
    CompoundPolicy, FileSandboxPolicy, NetworkSandboxPolicy,
)

agent = Agent.builder() \
    .permissions(CompoundPolicy([
        AllowReadsAskWrites(),
        FileSandboxPolicy(["/workspace", "/tmp"]),
        NetworkSandboxPolicy(["https://api.github.com/*"]),
    ])) \
    .build()
```

### Middleware

Pipeline-based middleware for cross-cutting concerns.

```python
from curio_agent_sdk import (
    CostTracker, GuardrailsMiddleware, RateLimitMiddleware,
)

agent = Agent(
    middleware=[
        CostTracker(
            budget=10.00,
            alert_thresholds=[0.5, 0.8, 0.95],
            on_threshold=lambda pct, cost: notify(f"At {pct*100}%: ${cost:.2f}"),
        ),
        GuardrailsMiddleware(
            block_patterns=[r"(?i)password"],
            block_input_patterns=[r"(?i)ignore previous"],
            block_prompt_injection=True,
        ),
        RateLimitMiddleware(requests_per_minute=60),
    ],
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
    elif event.type == "thinking":
        print(f"[Thinking: {event.text}]")
    elif event.type == "done":
        print("\n[Done]")
```

### Event Bus

Distributed event streaming with pub/sub, replay, and dead letter queues.

```python
from curio_agent_sdk import InMemoryEventBus

bus = InMemoryEventBus()
agent = Agent.builder().event_bus(bus).model("openai:gpt-4o").build()

# Subscribe with glob patterns
await bus.subscribe("tool.call.*", lambda e: print(f"Tool: {e.data}"))
await bus.subscribe("*.error", lambda e: print(f"Error: {e.data}"))
await bus.subscribe("*", audit_logger)

# Replay events from a timestamp
async for event in bus.replay(start_time, pattern="llm.call.*"):
    print(event.to_dict())

# Dead letter inspection
for entry in bus.dead_letters:
    print(f"Failed: {entry.handler} — {entry.error}")
```

### Plugins

Distributable plugin packages with auto-discovery via entry points.

```python
from curio_agent_sdk import Plugin

class MyPlugin(Plugin):
    name = "my-plugin"
    version = "1.0.0"

    def register(self, builder):
        builder.tool(my_tool)
        builder.hook("agent.run.before", my_hook)

agent = Agent.builder() \
    .plugin(MyPlugin()) \
    .discover_plugins()  # Auto-discover from installed packages
    .build()
```

### Checkpointing & Resume

```python
from curio_agent_sdk import FileStateStore

agent = Agent(state_store=FileStateStore("./state"), checkpoint_interval=1, ...)

result = await agent.arun("Long task...")
result = await agent.arun("Continue...", resume_from=result.run_id)
```

### Component Lifecycle

All stateful components implement `Component` with `startup()`, `shutdown()`, and `health_check()`. The agent manages lifecycle automatically.

```python
async with agent:
    result = await agent.arun("Do something")
    health = await agent.runtime.system_health()
    # {"healthy": True, "components": {"MemoryManager": True, "MCPBridge": True, ...}}
```

### CLI Harness

Build interactive CLI agents with streaming, commands, and session persistence.

```python
from curio_agent_sdk import AgentCLI

cli = AgentCLI(agent)
cli.register_command("/deploy", deploy_handler)
await cli.run_interactive()  # REPL with /help, /clear, /status, /sessions, /skills
```

### Testing

Comprehensive testing utilities: mocking, record/replay, snapshots, benchmarks, coverage, and evals.

```python
from curio_agent_sdk.testing import (
    MockLLM, AgentTestHarness, ToolTestKit,
    RecordingMiddleware, ReplayLLMClient,
    BenchmarkSuite, AgentEvalSuite, AgentCoverageTracker,
)

# Mock-based testing
mock = MockLLM()
mock.add_response(tool_call_response("calculate", {"expression": "2+2"}))
mock.add_text_response("2 + 2 = 4")

harness = AgentTestHarness(agent, llm=mock)
result = harness.run_sync("What is 2+2?")
assert result.status == "completed"
assert harness.tool_calls == [("calculate", {"expression": "2+2"})]

# Tool-level testing
kit = ToolTestKit()
kit.mock_tool("read_file", returns="file content")
kit.assert_tool_called("read_file", path="test.py")
kit.assert_call_order(["read_file", "write_file"])

# Record/replay (capture real runs, replay deterministically)
recorder = RecordingMiddleware()
recorder.save("tests/fixtures/run.json")
replay = ReplayLLMClient.from_file("tests/fixtures/run.json")

# Evals and regression detection
eval_suite = AgentEvalSuite(agent=agent, dataset=[...], metrics=[...])
results = await eval_suite.run()

# Pytest fixtures (add to conftest.py)
# pytest_plugins = ["curio_agent_sdk.testing.fixtures"]
# Provides: mock_llm, agent_test_harness, tool_test_kit, in_memory_state_store, etc.
```

## Reliability & Production

### Circuit Breakers

```python
from curio_agent_sdk import CircuitBreaker, TieredRouter, FallbackToLowerTier

router = TieredRouter(
    tier1=["groq:llama-3.1-8b-instant"],
    tier2=["openai:gpt-4o"],
    degradation_strategy=FallbackToLowerTier(),
)
```

### Request Deduplication

```python
client = LLMClient(router=router, dedup_enabled=True, dedup_ttl=30.0)
```

### Credential Management

```python
from curio_agent_sdk import VaultCredentialResolver, AWSSecretsResolver

agent = Agent.builder() \
    .credential_resolver(VaultCredentialResolver("https://vault:8200", token="...")) \
    .build()
```

### Observability

Hook-based observability consumers replace middleware for unified event-driven monitoring.

```python
from curio_agent_sdk import TracingConsumer, LoggingConsumer, PersistenceConsumer, PrometheusExporter

agent = Agent.builder() \
    .hook("llm.call.after", TracingConsumer(tracer)) \
    .hook("llm.call.after", LoggingConsumer(logger)) \
    .hook("*", PrometheusExporter()) \
    .build()
```

## Tier System

Three tiers for different task complexities with automatic failover:

| Tier | Purpose | Default Models |
|------|---------|----------------|
| tier1 | Fast, simple tasks | gpt-4o-mini, llama-3.1-8b |
| tier2 | Balanced quality/speed | gpt-4o, claude-sonnet-4-6 |
| tier3 | High-quality output | gpt-4o, claude-sonnet-4-6 |

## Observability Dashboard

A separate **[Curio Agent Observability](../curio_agent_observability/)** dashboard provides a web UI with:

- Agent run history and event timeline
- Cost tracking and per-model breakdowns
- Tool analytics and performance metrics
- Trace viewer for distributed agent runs

## License

Apache License 2.0
