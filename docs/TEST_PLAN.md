# Curio Agent SDK — Comprehensive Test Plan

> **Version:** 1.0
> **SDK Version:** 0.6.0
> **Last Updated:** 2026-02-28
> **Status:** Draft

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Test Infrastructure & Tooling](#2-test-infrastructure--tooling)
3. [Directory Structure](#3-directory-structure)
4. [Test Categories](#4-test-categories)
5. [Phase 1 — Foundation: Models, Exceptions & Base Classes](#5-phase-1--foundation-models-exceptions--base-classes) ✅
6. [Phase 2 — Core Tooling System](#6-phase-2--core-tooling-system) ✅
7. [Phase 3 — LLM Client & Provider Layer](#7-phase-3--llm-client--provider-layer) ✅
8. [Phase 4 — Agent Loop & Runtime](#8-phase-4--agent-loop--runtime)
9. [Phase 5 — State, Checkpoint & Session Management](#9-phase-5--state-checkpoint--session-management)
10. [Phase 6 — Memory System](#10-phase-6--memory-system)
11. [Phase 7 — Events, Hooks & Middleware](#11-phase-7--events-hooks--middleware)
12. [Phase 8 — Security & Permissions](#12-phase-8--security--permissions)
13. [Phase 9 — Extensions: Skills, Subagents & Plugins](#13-phase-9--extensions-skills-subagents--plugins)
14. [Phase 10 — MCP & Connectors](#14-phase-10--mcp--connectors)
15. [Phase 11 — Workflow: Plan Mode & Structured Output](#15-phase-11--workflow-plan-mode--structured-output)
16. [Phase 12 — Persistence Layer](#16-phase-12--persistence-layer)
17. [Phase 13 — Built-in Tools](#17-phase-13--built-in-tools)
18. [Phase 14 — Context & Credentials](#18-phase-14--context--credentials)
19. [Phase 15 — CLI](#19-phase-15--cli)
20. [Phase 16 — Testing Utilities (Meta-Tests)](#20-phase-16--testing-utilities-meta-tests)
21. [Phase 17 — Integration Tests](#21-phase-17--integration-tests)
22. [Phase 18 — End-to-End / Example Tests](#22-phase-18--end-to-end--example-tests)
23. [Phase 19 — Performance & Stress Tests](#23-phase-19--performance--stress-tests)
24. [Phase 20 — CI/CD & Coverage Configuration](#24-phase-20--cicd--coverage-configuration)
25. [Running Tests](#25-running-tests)
26. [Coverage Targets](#26-coverage-targets)
27. [Test Conventions & Best Practices](#27-test-conventions--best-practices)

---

## 1. Overview & Goals

### Purpose
This plan defines a comprehensive testing strategy for the Curio Agent SDK, covering every module, every public API, every edge case, and every integration point. The goal is to achieve **maximum code coverage and confidence** so that any regression, behavioral change, or broken contract is caught immediately.

### Goals
- **Unit test coverage ≥ 90%** for all modules
- **Integration test coverage** for all cross-module interactions
- **End-to-end examples** that validate the SDK works as documented
- **Performance baselines** for critical paths (agent loop, LLM routing, tool execution)
- **Regression safety net** — any commit that breaks existing behavior is caught in CI
- **Documentation-as-tests** — examples from the README are runnable tests

### Non-Goals
- Testing third-party libraries themselves (openai, anthropic SDKs)
- Testing real LLM API calls in CI (these are behind optional markers)
- Load testing / scalability benchmarks (separate effort)

---

## 2. Test Infrastructure & Tooling

### Core Dependencies (add to `[project.optional-dependencies]` in `pyproject.toml`)

```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "pytest-xdist>=3.5.0",      # Parallel test execution
    "pytest-timeout>=2.2.0",    # Timeout enforcement
    "pytest-mock>=3.12.0",      # Mocker fixture
    "coverage[toml]>=7.4.0",    # Coverage with pyproject.toml config
    "aiosqlite>=0.19.0",        # Async SQLite for persistence tests
    "pydantic>=2.0.0",          # Structured output tests
]
```

### Configuration (add to `pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests (fast, no external deps)",
    "integration: Integration tests (cross-module, may use filesystem)",
    "e2e: End-to-end tests (full agent runs with mock LLM)",
    "slow: Tests that take >5 seconds",
    "live: Tests requiring real API keys (skipped in CI by default)",
    "examples: Tests that validate SDK examples",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
timeout = 30

[tool.coverage.run]
source = ["src/curio_agent_sdk"]
branch = true
omit = [
    "src/curio_agent_sdk/tools/computer_use.py",
    "src/curio_agent_sdk/tools/browser.py",
    "src/curio_agent_sdk/persistence/postgres.py",
]

[tool.coverage.report]
show_missing = true
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__",
    "@abstractmethod",
    "raise NotImplementedError",
    "\\.\\.\\.",
]

[tool.coverage.html]
directory = "htmlcov"
```

### Shared Fixtures (`tests/conftest.py`)

The root `conftest.py` will provide:
- `mock_llm` — Pre-configured `MockLLM` instance
- `sample_tool` — A simple `@tool` decorated function
- `sample_agent` — A minimal `Agent` with `MockLLM`
- `sample_messages` — Common message sequences
- `tmp_dir` — Temporary directory (via `tmp_path`)
- `event_collector` — Hook handler that records all events
- `text_response_factory` — Factory for `LLMResponse` objects
- `tool_call_response_factory` — Factory for tool-call `LLMResponse` objects

---

## 3. Directory Structure

```
tests/
├── conftest.py                         # Shared fixtures, markers, helpers
├── helpers/                            # Shared test utilities
│   ├── __init__.py
│   ├── factories.py                    # Object factories for test data
│   ├── assertions.py                   # Custom assertion helpers
│   └── mocks.py                        # Reusable mock objects
│
├── unit/                               # Unit tests (fast, isolated)
│   ├── conftest.py
│   ├── models/
│   │   ├── test_llm_models.py          # Message, ToolCall, TokenUsage, LLMRequest, LLMResponse
│   │   ├── test_agent_models.py        # AgentRun, AgentRunResult, AgentRunStatus
│   │   ├── test_event_models.py        # EventType, StreamEvent, AgentEvent
│   │   └── test_exceptions.py          # Exception hierarchy
│   ├── base/
│   │   └── test_component.py           # Component ABC lifecycle
│   ├── tools/
│   │   ├── test_tool.py                # Tool class, @tool decorator, ToolConfig
│   │   ├── test_tool_schema.py         # ToolSchema, ToolParameter, from_function
│   │   ├── test_tool_registry.py       # ToolRegistry register/get/remove
│   │   └── test_tool_executor.py       # ToolExecutor execution, ToolResult
│   ├── llm/
│   │   ├── test_llm_client.py          # LLMClient call/stream/batch
│   │   ├── test_tiered_router.py       # TieredRouter routing, degradation
│   │   ├── test_token_counter.py       # Token counting, caching
│   │   ├── test_batch_client.py        # BatchLLMClient
│   │   └── providers/
│   │       ├── test_openai_provider.py
│   │       ├── test_anthropic_provider.py
│   │       ├── test_groq_provider.py
│   │       └── test_ollama_provider.py
│   ├── agent/
│   │   ├── test_agent.py               # Agent construction, run, arun
│   │   ├── test_agent_builder.py       # AgentBuilder fluent API
│   │   └── test_runtime.py             # Runtime orchestration
│   ├── state/
│   │   ├── test_agent_state.py         # AgentState operations, extensions
│   │   ├── test_state_store.py         # InMemoryStateStore, FileStateStore
│   │   ├── test_checkpoint.py          # Checkpoint serialize/deserialize
│   │   └── test_session.py             # Session, SessionStore, SessionManager
│   ├── memory/
│   │   ├── test_memory_base.py         # Memory ABC contract tests
│   │   ├── test_conversation_memory.py
│   │   ├── test_key_value_memory.py
│   │   ├── test_working_memory.py
│   │   ├── test_episodic_memory.py
│   │   ├── test_graph_memory.py
│   │   ├── test_self_editing_memory.py
│   │   ├── test_file_memory.py
│   │   ├── test_composite_memory.py
│   │   ├── test_vector_memory.py
│   │   ├── test_memory_manager.py      # MemoryManager + strategies
│   │   └── test_memory_policies.py     # Decay, importance policies
│   ├── events/
│   │   ├── test_hook_registry.py       # HookRegistry on/off/emit, priority
│   │   ├── test_hook_context.py        # HookContext cancel/modify
│   │   └── test_event_bus.py           # EventBus pub/sub, replay, dead letters
│   ├── middleware/
│   │   ├── test_middleware_base.py      # Middleware ABC, MiddlewarePipeline
│   │   ├── test_logging_mw.py
│   │   ├── test_cost_tracker.py
│   │   ├── test_rate_limit.py
│   │   ├── test_tracing.py
│   │   ├── test_guardrails.py
│   │   ├── test_consumers.py
│   │   └── test_prometheus.py
│   ├── security/
│   │   ├── test_permissions.py         # All PermissionPolicy implementations
│   │   └── test_human_input.py         # HumanInputHandler
│   ├── extensions/
│   │   ├── test_skills.py              # Skill, SkillRegistry
│   │   ├── test_subagent.py            # SubagentConfig, AgentOrchestrator
│   │   └── test_plugins.py            # Plugin, discovery
│   ├── loops/
│   │   ├── test_agent_loop.py          # AgentLoop ABC contract
│   │   └── test_tool_calling_loop.py   # ToolCallingLoop step/should_continue
│   ├── workflow/
│   │   ├── test_plan_mode.py           # PlanMode, PlanState, Plan, TodoManager
│   │   └── test_structured_output.py   # response_format_to_schema, parsing
│   ├── context/
│   │   ├── test_context_manager.py     # ContextManager token budgets
│   │   └── test_instruction_loader.py  # InstructionLoader
│   ├── credentials/
│   │   └── test_credentials.py         # Env, Vault, AWS resolvers
│   ├── mcp/
│   │   ├── test_mcp_client.py          # MCPClient connect/list/call
│   │   ├── test_mcp_transport.py       # Transport implementations
│   │   ├── test_mcp_config.py          # MCPServerConfig
│   │   ├── test_mcp_adapter.py         # MCPToolAdapter
│   │   └── test_mcp_bridge.py          # MCPBridge lifecycle
│   ├── connectors/
│   │   ├── test_connector.py           # Connector ABC
│   │   └── test_connector_bridge.py    # ConnectorBridge lifecycle
│   ├── persistence/
│   │   ├── test_base_persistence.py    # BasePersistence ABC
│   │   ├── test_sqlite_persistence.py  # SQLitePersistence
│   │   └── test_memory_persistence.py  # InMemoryPersistence
│   ├── built_in_tools/
│   │   ├── test_file_tools.py          # file_read, file_write
│   │   ├── test_code_tools.py          # python_execute, shell_execute
│   │   ├── test_web_tools.py           # web_fetch
│   │   └── test_http_tools.py          # http_request
│   ├── cli/
│   │   └── test_cli.py                 # AgentCLI
│   ├── resilience/
│   │   └── test_circuit_breaker.py     # CircuitBreaker
│   └── testing/                        # Meta-tests: test the testing utilities
│       ├── test_mock_llm.py
│       ├── test_harness.py
│       ├── test_toolkit.py
│       ├── test_integration_harness.py
│       ├── test_replay.py
│       ├── test_snapshot.py
│       ├── test_benchmark.py
│       ├── test_eval.py
│       ├── test_regression.py
│       └── test_coverage_tracker.py
│
├── integration/                        # Integration tests (cross-module)
│   ├── conftest.py
│   ├── test_agent_with_tools.py        # Agent + Tool chain execution
│   ├── test_agent_with_memory.py       # Agent + MemoryManager end-to-end
│   ├── test_agent_with_middleware.py   # Agent + Middleware pipeline
│   ├── test_agent_with_hooks.py        # Agent + Hook lifecycle
│   ├── test_agent_with_state.py        # Agent + State/Checkpoint flow
│   ├── test_agent_with_sessions.py     # Agent + SessionManager multi-turn
│   ├── test_agent_with_permissions.py  # Agent + PermissionPolicy enforcement
│   ├── test_agent_with_skills.py       # Agent + SkillRegistry
│   ├── test_agent_with_subagents.py    # Agent + AgentOrchestrator
│   ├── test_agent_with_plan_mode.py    # Agent + PlanMode workflow
│   ├── test_agent_with_structured.py   # Agent + Structured output (Pydantic)
│   ├── test_agent_streaming.py         # Agent streaming events
│   ├── test_agent_builder_full.py      # Full builder chain → run
│   ├── test_middleware_pipeline.py     # Multiple middleware stacked
│   ├── test_memory_persistence.py      # Memory + Persistence backends
│   ├── test_llm_routing.py            # LLMClient + Router + Providers
│   ├── test_mcp_integration.py         # MCPClient + MCPBridge + Tool adapter
│   ├── test_connector_integration.py   # Connector + ConnectorBridge + Agent
│   ├── test_plugin_system.py           # Plugin discovery + registration
│   ├── test_event_bus_integration.py   # EventBus + Agent lifecycle
│   └── test_cost_budget.py            # CostTracker budget enforcement
│
├── e2e/                                # End-to-end example tests
│   ├── conftest.py
│   ├── test_simple_agent.py            # Basic agent with text response
│   ├── test_tool_agent.py              # Agent that calls tools
│   ├── test_multi_turn_agent.py        # Multi-turn conversation
│   ├── test_coding_agent.py            # Agent with file/code tools
│   ├── test_research_agent.py          # Agent with web/http tools
│   ├── test_multi_agent.py             # Multi-agent orchestration
│   ├── test_resilient_agent.py         # Timeout, retry, degradation
│   ├── test_memory_agent.py            # Agent with persistent memory
│   └── test_structured_agent.py        # Agent with Pydantic output
│
├── performance/                        # Performance & stress tests
│   ├── conftest.py
│   ├── test_tool_execution_perf.py     # Tool execution throughput
│   ├── test_memory_operations_perf.py  # Memory read/write at scale
│   ├── test_state_checkpoint_perf.py   # Checkpoint serialization speed
│   └── test_middleware_overhead.py     # Middleware pipeline overhead
│
└── live/                               # Live API tests (require API keys)
    ├── conftest.py                     # Skip if no API keys
    ├── test_openai_live.py
    ├── test_anthropic_live.py
    ├── test_groq_live.py
    └── test_ollama_live.py
```

---

## 4. Test Categories

| Category | Marker | Purpose | Speed | External Deps |
|----------|--------|---------|-------|---------------|
| **Unit** | `@pytest.mark.unit` | Test individual classes/functions in isolation | Fast (<1s each) | None |
| **Integration** | `@pytest.mark.integration` | Test cross-module interactions | Medium (<5s each) | Filesystem only |
| **E2E** | `@pytest.mark.e2e` | Validate complete workflows | Medium (<10s each) | Filesystem, MockLLM |
| **Performance** | `@pytest.mark.slow` | Benchmark critical paths | Slow (>5s) | None |
| **Live** | `@pytest.mark.live` | Test against real APIs | Slow, flaky | API keys required |

---

## 5. Phase 1 — Foundation: Models, Exceptions & Base Classes ✅

**Priority:** Highest (everything depends on these)
**Estimated tests:** ~80 → **206 tests written, 206 passed, 100% coverage**
**Status:** ✅ COMPLETED

### 5.1 `models/llm.py` — Data Models ✅

**File:** `tests/unit/models/test_llm_models.py` — **57 tests, all passing**

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_message_creation_user` | `Message(role="user", content="hello")` creates correctly | ✅ |
| 2 | `test_message_creation_assistant` | Assistant message with content | ✅ |
| 3 | `test_message_creation_system` | System message with content | ✅ |
| 4 | `test_message_creation_tool` | Tool message with `tool_call_id` | ✅ |
| 5 | `test_message_with_tool_calls` | Message with `tool_calls` list | ✅ |
| 6 | `test_message_with_content_blocks` | Message with list of `ContentBlock` | ✅ |
| 7 | `test_message_defaults` | Default values (None content, no tool_calls) | ✅ |
| 8 | `test_tool_call_creation` | `ToolCall(id, name, arguments)` | ✅ |
| 9 | `test_tool_call_with_complex_args` | Nested dict/list arguments | ✅ |
| 10 | `test_token_usage_defaults` | All zeros by default | ✅ |
| 11 | `test_token_usage_total` | `input_tokens + output_tokens` calculation | ✅ |
| 12 | `test_token_usage_with_cache` | Cache read/write token tracking | ✅ |
| 13 | `test_llm_request_minimal` | Only required fields | ✅ |
| 14 | `test_llm_request_full` | All fields populated | ✅ |
| 15 | `test_llm_request_defaults` | Default values for optional fields | ✅ |
| 16 | `test_llm_request_with_response_format` | `response_format` dict | ✅ |
| 17 | `test_llm_request_with_metadata` | Custom metadata dict | ✅ |
| 18 | `test_llm_response_creation` | Full `LLMResponse` construction | ✅ |
| 19 | `test_llm_response_error` | Response with error field | ✅ |
| 20 | `test_llm_response_finish_reasons` | "stop", "tool_use", "length", "error" | ✅ |
| + | Additional: ContentBlock, ToolSchema, LLMStreamChunk, Message factories, text property | Full coverage of all classes and properties | ✅ |

### 5.2 `models/agent.py` — Agent Models ✅

**File:** `tests/unit/models/test_agent_models.py` — **36 tests, all passing**

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_agent_run_status_enum` | All enum values: PENDING, RUNNING, COMPLETED, ERROR, CANCELLED, TIMEOUT | ✅ |
| 2 | `test_agent_run_result_defaults` | Default field values | ✅ |
| 3 | `test_agent_run_result_completed` | Status="completed" with output | ✅ |
| 4 | `test_agent_run_result_error` | Status="error" with error message | ✅ |
| 5 | `test_agent_run_result_with_parsed_output` | `parsed_output` set for structured output | ✅ |
| 6 | `test_agent_run_result_metrics` | Token/iteration/call counts | ✅ |
| 7 | `test_agent_run_creation` | `AgentRun` with all fields | ✅ |
| 8 | `test_agent_run_defaults` | Default field values | ✅ |
| 9 | `test_agent_run_timing` | `started_at` and `finished_at` fields | ✅ |
| + | Additional: to_dict, from_dict, roundtrip, AgentRunEvent, AgentLLMUsage, is_success, get_data_dict | Full coverage of all classes, serialization, and edge cases | ✅ |

### 5.3 `models/events.py` — Event Models ✅

**File:** `tests/unit/models/test_event_models.py` — **25 tests, all passing**

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_event_type_enum_values` | All EventType values exist (run, iteration, LLM, tool, phases, state, custom) | ✅ |
| 2 | `test_stream_event_text_delta` | `type="text_delta"` with `text` field | ✅ |
| 3 | `test_stream_event_tool_call_start` | `type="tool_call_start"` with `tool_name`, `tool_args` | ✅ |
| 4 | `test_stream_event_tool_call_end` | `type="tool_call_end"` with `tool_result` | ✅ |
| 5 | `test_stream_event_error` | `type="error"` with `error` field | ✅ |
| 6 | `test_stream_event_done` | `type="done"` event | ✅ |
| 7 | `test_stream_event_thinking` | `type="thinking"` event | ✅ |
| 8 | `test_stream_event_iteration` | `type="iteration_start"/"iteration_end"` with iteration number | ✅ |
| 9 | `test_agent_event_creation` | `AgentEvent` dataclass | ✅ |
| + | Additional: to_dict, timestamp format, data independence, defaults, str enum | Full coverage of all event types and edge cases | ✅ |

### 5.4 `models/exceptions.py` — Exception Hierarchy ✅

**File:** `tests/unit/models/test_exceptions.py` — **77 tests, all passing**

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_curio_error_is_base` | `CurioError` extends `Exception` | ✅ |
| 2 | `test_llm_error_hierarchy` | `LLMError → CurioError` | ✅ |
| 3 | `test_llm_rate_limit_error` | `LLMRateLimitError → LLMError → CurioError` | ✅ |
| 4 | `test_llm_authentication_error` | Correct hierarchy | ✅ |
| 5 | `test_llm_provider_error` | Correct hierarchy + status_code | ✅ |
| 6 | `test_llm_timeout_error` | Correct hierarchy | ✅ |
| 7 | `test_no_available_model_error` | Correct hierarchy | ✅ |
| 8 | `test_cost_budget_exceeded` | Correct hierarchy + total_cost/budget | ✅ |
| 9 | `test_tool_error_hierarchy` | `ToolError → CurioError` | ✅ |
| 10 | `test_tool_not_found_error` | Correct hierarchy + available list | ✅ |
| 11 | `test_tool_execution_error` | Correct hierarchy + cause | ✅ |
| 12 | `test_tool_timeout_error` | Correct hierarchy + timeout value | ✅ |
| 13 | `test_tool_validation_error` | Correct hierarchy + errors list | ✅ |
| 14 | `test_agent_error_hierarchy` | `AgentError → CurioError` | ✅ |
| 15 | `test_agent_timeout_error` | Correct hierarchy + timeout/iterations | ✅ |
| 16 | `test_agent_cancelled_error` | Correct hierarchy | ✅ |
| 17 | `test_max_iterations_error` | Correct hierarchy + max_iterations | ✅ |
| 18 | `test_config_error` | `ConfigError → CurioError` | ✅ |
| 19 | `test_exceptions_are_catchable_by_base` | `try/except CurioError` catches all (17 parametrized) | ✅ |
| 20 | `test_exception_messages` | Error messages are preserved | ✅ |
| + | Additional: parametrized catch-all by LLMError, ToolError, AgentError; attribute tests, defaults, edge cases | Full hierarchy coverage with cross-cutting catch tests | ✅ |

### 5.5 `base/component.py` — Component ABC ✅

**File:** `tests/unit/base/test_component.py` — **11 tests, all passing**

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_component_is_abstract` | Component is an ABC | ✅ |
| 2 | `test_can_subclass` | Can create concrete subclass | ✅ |
| 3 | `test_startup_default` | Default `startup()` does nothing | ✅ |
| 4 | `test_shutdown_default` | Default `shutdown()` does nothing | ✅ |
| 5 | `test_health_check_default` | Default `health_check()` returns True | ✅ |
| 6 | `test_custom_startup` | Subclass can override startup | ✅ |
| 7 | `test_custom_shutdown` | Subclass can override shutdown | ✅ |
| 8 | `test_custom_health_check` | Subclass can override health_check | ✅ |
| 9 | `test_lifecycle_order` | startup → use → shutdown in correct order | ✅ |
| 10 | `test_startup_idempotent` | Multiple startup() calls are safe | ✅ |
| 11 | `test_shutdown_idempotent` | Multiple shutdown() calls are safe | ✅ |

---

## 6. Phase 2 — Core Tooling System ✅

**Priority:** Very High
**Estimated tests:** ~70 → **Actual: 129 tests (87% coverage)**

### 6.1 `core/tools/tool.py` — Tool & @tool Decorator ✅ (34 tests, 98% coverage)

**File:** `tests/unit/tools/test_tool.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_tool_from_function` | Wrap a sync function as Tool |
| 2 | `test_tool_from_async_function` | Wrap an async function as Tool |
| 3 | `test_tool_name_from_function` | Name derived from function name |
| 4 | `test_tool_name_override` | Explicit name overrides function name |
| 5 | `test_tool_description_from_docstring` | Description from docstring |
| 6 | `test_tool_description_override` | Explicit description |
| 7 | `test_tool_execute_sync` | Execute a sync tool |
| 8 | `test_tool_execute_async` | Execute an async tool |
| 9 | `test_tool_execute_with_args` | Pass keyword arguments |
| 10 | `test_tool_execute_error_handling` | Tool raising an exception |
| 11 | `test_tool_config_defaults` | Default ToolConfig values |
| 12 | `test_tool_config_custom` | Custom timeout, retries, etc. |
| 13 | `test_tool_decorator_basic` | `@tool` with no args |
| 14 | `test_tool_decorator_with_name` | `@tool(name="custom")` |
| 15 | `test_tool_decorator_with_config` | `@tool(timeout=30, max_retries=3)` |
| 16 | `test_tool_decorator_preserves_function` | Decorated function still works normally |
| 17 | `test_tool_with_type_hints` | Type hints are used for schema |
| 18 | `test_tool_with_default_args` | Default values reflected in schema |
| 19 | `test_tool_timeout_enforcement` | Tool execution respects timeout |
| 20 | `test_tool_retry_on_failure` | Tool retries on error (max_retries > 0) |

### 6.2 `core/tools/schema.py` — ToolSchema & ToolParameter ✅ (49 tests, 95% coverage)

**File:** `tests/unit/tools/test_tool_schema.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_tool_parameter_string` | ToolParameter with type="string" |
| 2 | `test_tool_parameter_integer` | ToolParameter with type="integer" |
| 3 | `test_tool_parameter_boolean` | ToolParameter with type="boolean" |
| 4 | `test_tool_parameter_array` | ToolParameter with type="array" and items |
| 5 | `test_tool_parameter_enum` | ToolParameter with enum constraint |
| 6 | `test_tool_parameter_optional` | `required=False` with default |
| 7 | `test_tool_parameter_to_json_schema` | JSON schema generation |
| 8 | `test_tool_schema_creation` | Basic ToolSchema |
| 9 | `test_tool_schema_to_json_schema` | Full JSON schema output |
| 10 | `test_tool_schema_to_llm_schema` | LLM-compatible schema format |
| 11 | `test_tool_schema_validate_valid_args` | Validation passes |
| 12 | `test_tool_schema_validate_missing_required` | Validation fails for missing required |
| 13 | `test_tool_schema_validate_extra_args` | Extra args handling |
| 14 | `test_tool_schema_from_function_simple` | Infer schema from simple function |
| 15 | `test_tool_schema_from_function_complex` | Function with various param types |
| 16 | `test_tool_schema_from_function_with_docstring` | Description from docstring |
| 17 | `test_tool_schema_from_function_no_hints` | Function without type hints |

### 6.3 `core/tools/registry.py` — ToolRegistry ✅ (22 tests, 100% coverage)

**File:** `tests/unit/tools/test_tool_registry.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_registry_register_tool` | Register a Tool object |
| 2 | `test_registry_register_function` | Register a plain function |
| 3 | `test_registry_register_decorated` | Register a @tool decorated function |
| 4 | `test_registry_get_existing` | Get a registered tool by name |
| 5 | `test_registry_get_nonexistent` | Raises ToolNotFoundError |
| 6 | `test_registry_has_tool` | `has()` returns True/False |
| 7 | `test_registry_remove_tool` | Remove and verify gone |
| 8 | `test_registry_remove_nonexistent` | Returns False |
| 9 | `test_registry_list_tools` | `.tools` property returns all |
| 10 | `test_registry_list_names` | `.names` property returns all names |
| 11 | `test_registry_duplicate_name` | Behavior on duplicate registration |
| 12 | `test_registry_get_llm_schemas` | Get schemas for all tools |
| 13 | `test_registry_empty` | Empty registry behavior |
| 14 | `test_registry_init_with_tools` | Initialize with tool list |

### 6.4 `core/tools/executor.py` — ToolExecutor & ToolResult ✅ (24 tests, 73% coverage)

**File:** `tests/unit/tools/test_tool_executor.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_tool_result_success` | ToolResult with no error |
| 2 | `test_tool_result_error` | ToolResult with error string |
| 3 | `test_tool_result_is_error` | `is_error` property |
| 4 | `test_tool_result_to_message` | Convert to Message |
| 5 | `test_executor_execute_success` | Execute a tool successfully |
| 6 | `test_executor_execute_not_found` | Tool not in registry |
| 7 | `test_executor_execute_error` | Tool raises exception |
| 8 | `test_executor_execute_parallel` | Execute multiple tools in parallel |
| 9 | `test_executor_execute_parallel_partial_failure` | Some tools fail, others succeed |
| 10 | `test_executor_with_permission_policy_allow` | PermissionPolicy allows |
| 11 | `test_executor_with_permission_policy_deny` | PermissionPolicy denies |
| 12 | `test_executor_with_permission_policy_ask` | PermissionPolicy asks, human approves |
| 13 | `test_executor_with_permission_policy_ask_denied` | PermissionPolicy asks, human denies |
| 14 | `test_executor_with_hooks` | Hook emitted before/after tool call |
| 15 | `test_executor_hook_cancellation` | Hook cancels tool execution |

---

## 7. Phase 3 — LLM Client & Provider Layer ✅

**Priority:** Very High
**Estimated tests:** ~60
**Status:** ✅ Complete (92 tests passing)

### 7.1 `core/llm/client.py` — LLMClient

**File:** `tests/unit/llm/test_llm_client.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_client_call_basic` | Basic call with mocked provider | ✅ |
| 2 | `test_client_call_with_tools` | Call with tool schemas | ✅ |
| 3 | `test_client_call_routing` | Request routed to correct provider | ✅ |
| 4 | `test_client_call_with_middleware` | Middleware pipeline executed | ✅ |
| 5 | `test_client_call_error_handling` | Provider error bubbles up | ✅ |
| 6 | `test_client_stream_basic` | Stream response with mocked provider | ✅ |
| 7 | `test_client_batch_basic` | Batch multiple requests | ✅ |
| 8 | `test_client_dedup_enabled` | Deduplication prevents duplicate calls | ✅ |
| 9 | `test_client_dedup_disabled` | No dedup when disabled | ✅ |
| 10 | `test_client_startup_shutdown` | Component lifecycle | ✅ |
| 11 | `test_client_custom_providers` | Register custom provider | ✅ |
| 12 | `test_client_fallback_on_error` | Falls back to alternative provider | ✅ |

### 7.2 `core/llm/router.py` — TieredRouter

**File:** `tests/unit/llm/test_tiered_router.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_router_route_tier1` | Routes to tier1 provider | ✅ |
| 2 | `test_router_route_tier2` | Routes to tier2 provider | ✅ |
| 3 | `test_router_route_tier3` | Routes to tier3 provider | ✅ |
| 4 | `test_router_round_robin` | Rotates among providers in same tier | ✅ |
| 5 | `test_router_unhealthy_provider_skip` | Skips unhealthy providers | ✅ |
| 6 | `test_router_all_unhealthy` | All providers down → degradation strategy | ✅ |
| 7 | `test_router_reset_and_retry_strategy` | ResetAndRetry behavior | ✅ |
| 8 | `test_router_fallback_to_lower_tier` | FallbackToLowerTier behavior | ✅ |
| 9 | `test_router_raise_error_strategy` | RaiseError behavior | ✅ |
| 10 | `test_router_empty_tier` | No providers in tier | ✅ |
| 11 | `test_router_provider_health_tracking` | Mark provider healthy/unhealthy | ✅ |
| 12 | `test_router_auto_detection` | Auto-detect from environment variables | ✅ |

### 7.3 `core/llm/token_counter.py` — Token Counting

**File:** `tests/unit/llm/test_token_counter.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_count_tokens_simple` | Count tokens in a string | ✅ |
| 2 | `test_count_tokens_messages` | Count tokens across messages | ✅ |
| 3 | `test_count_tokens_with_tools` | Count includes tool schemas | ✅ |
| 4 | `test_count_tokens_caching` | Cached results returned | ✅ |
| 5 | `test_count_tokens_different_models` | Different models = different counts | ✅ |
| 6 | `test_count_tokens_empty` | Empty message list | ✅ |
| 7 | `test_fallback_estimation` | When tiktoken not installed | ✅ |

### 7.4 LLM Providers (unit tests with mocked HTTP)

**Files:** `tests/unit/llm/providers/test_*.py`

For each provider (OpenAI, Anthropic, Groq, Ollama):

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_provider_name` | `provider_name` attribute | ✅ |
| 2 | `test_provider_call_success` | Successful API call (mocked) | ✅ |
| 3 | `test_provider_call_with_tools` | Tool-use request formatting | ✅ |
| 4 | `test_provider_call_rate_limit` | 429 → `LLMRateLimitError` | ✅ |
| 5 | `test_provider_call_auth_error` | 401 → `LLMAuthenticationError` | ✅ |
| 6 | `test_provider_call_server_error` | 500 → `LLMProviderError` | ✅ |
| 7 | `test_provider_call_timeout` | Timeout → `LLMTimeoutError` | ✅ |
| 8 | `test_provider_stream` | Streaming response (mocked) | ✅ |
| 9 | `test_provider_request_formatting` | Request format matches API spec | ✅ |
| 10 | `test_provider_response_parsing` | Response parsed into LLMResponse | ✅ |

---

## 8. Phase 4 — Agent Loop & Runtime

**Priority:** Very High
**Estimated tests:** ~50
**Status:** ✅ Complete (78 tests passing)

### 8.1 `core/loops/base.py` — AgentLoop ABC

**File:** `tests/unit/loops/test_agent_loop.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_agent_loop_is_abstract` | Cannot instantiate directly | ✅ |
| 2 | `test_agent_loop_step_abstract` | `step()` must be implemented | ✅ |
| 3 | `test_agent_loop_should_continue_abstract` | `should_continue()` must be implemented | ✅ |
| 4 | `test_agent_loop_get_output_default` | Default `get_output()` extracts last assistant message | ✅ |
| 5 | `test_agent_loop_stream_step_default` | Default stream_step raises NotImplementedError or works | ✅ |
| 6 | `test_agent_loop_get_output_empty` | Empty messages returns empty string | ✅ |

### 8.2 `core/loops/tool_calling.py` — ToolCallingLoop

**File:** `tests/unit/loops/test_tool_calling_loop.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_loop_step_text_response` | LLM returns text → state updated, done | ✅ |
| 2 | `test_loop_step_tool_call` | LLM returns tool call → tool executed, result added | ✅ |
| 3 | `test_loop_step_multiple_tool_calls` | Multiple tool calls in one step | ✅ |
| 4 | `test_loop_step_parallel_tool_calls` | `parallel_tool_calls=True` | ✅ |
| 5 | `test_loop_step_sequential_tool_calls` | `parallel_tool_calls=False` | ✅ |
| 6 | `test_loop_should_continue_stop` | `finish_reason="stop"` → False | ✅ |
| 7 | `test_loop_should_continue_tool_use` | `finish_reason="tool_use"` → True | ✅ |
| 8 | `test_loop_should_continue_max_iterations` | At max → False | ✅ |
| 9 | `test_loop_should_continue_cancelled` | Cancel event set → False | ✅ |
| 10 | `test_loop_context_manager_integration` | Context window management | ✅ |
| 11 | `test_loop_with_response_format` | Structured output mode | ✅ |
| 12 | `test_loop_error_in_llm_call` | LLM error propagation | ✅ |
| 13 | `test_loop_error_in_tool_execution` | Tool error → error result message | ✅ |
| 14 | `test_loop_no_llm_raises` | Step without LLM raises RuntimeError | ✅ |

### 8.3 `core/agent/runtime.py` — Runtime

**File:** `tests/unit/agent/test_runtime.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_runtime_run_simple` | Single iteration, text response | ✅ |
| 2 | `test_runtime_run_with_tool_call` | Tool call → tool result → final text | ✅ |
| 3 | `test_runtime_run_multi_iteration` | Multiple loop iterations | ✅ |
| 4 | `test_runtime_run_max_iterations` | Hits max iterations limit | ✅ |
| 5 | `test_runtime_run_timeout` | Hits timeout | ✅ |
| 6 | `test_runtime_run_cancellation` | Cancel during execution | ✅ |
| 7 | `test_runtime_run_error` | LLM error during run | ✅ |
| 8 | `test_runtime_memory_injection` | Memory injected at start | ✅ |
| 9 | `test_runtime_memory_save` | Memory saved after run | ✅ |
| 10 | `test_runtime_hook_lifecycle` | All hooks emitted in order | ✅ |
| 11 | `test_runtime_checkpoint_save` | State checkpointed after iterations | ✅ |
| 12 | `test_runtime_create_state` | `create_state()` helper | ✅ |
| 13 | `test_runtime_create_state_with_context` | `create_state()` with context dict | ✅ |
| 14 | `test_runtime_run_with_state` | `run_with_state()` with pre-built state | ✅ |
| 15 | `test_runtime_startup_shutdown` | Component lifecycle | ✅ |
| 16 | `test_runtime_streaming` | `astream()` yields events | ✅ |

### 8.4 `core/agent/agent.py` — Agent

**File:** `tests/unit/agent/test_agent.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_agent_creation_minimal` | Minimal constructor | ✅ |
| 2 | `test_agent_creation_full` | All parameters | ✅ |
| 3 | `test_agent_arun` | `arun()` async | ✅ |
| 4 | `test_agent_astream` | `astream()` yields events | ✅ |
| 5 | `test_agent_context_manager` | `async with` lifecycle | ✅ |
| 6 | `test_agent_invoke_skill` | `invoke_skill()` method | ✅ |
| 7 | `test_agent_spawn_subagent` | `spawn_subagent()` method | ✅ |
| 8 | `test_agent_builder_classmethod` | `Agent.builder()` returns AgentBuilder | ✅ |
| 9 | `test_agent_id_generation` | Auto-generated agent_id | ✅ |
| 10 | `test_agent_custom_id` | Custom agent_id preserved | ✅ |
| 11 | `test_agent_tools_property` | tools property returns registered tools | ✅ |
| 12 | `test_agent_repr` | __repr__ includes id and name | ✅ |

### 8.5 `core/agent/builder.py` — AgentBuilder

**File:** `tests/unit/agent/test_agent_builder.py`

| # | Test Case | What It Validates | Status |
|---|-----------|-------------------|--------|
| 1 | `test_builder_system_prompt` | `.system_prompt()` sets prompt | ✅ |
| 2 | `test_builder_model` | `.model()` sets model | ✅ |
| 3 | `test_builder_tier` | `.tier()` sets tier | ✅ |
| 4 | `test_builder_llm` | `.llm()` sets client | ✅ |
| 5 | `test_builder_loop` | `.loop()` sets loop | ✅ |
| 6 | `test_builder_tools` | `.tools()` sets tool list | ✅ |
| 7 | `test_builder_tool_single` | `.add_tool()` adds one tool | ✅ |
| 8 | `test_builder_max_iterations` | `.max_iterations()` sets limit | ✅ |
| 9 | `test_builder_timeout` | `.timeout()` sets timeout | ✅ |
| 10 | `test_builder_temperature` | `.temperature()` sets temp | ✅ |
| 11 | `test_builder_middleware` | `.middleware()` sets pipeline | ✅ |
| 12 | `test_builder_human_input` | `.human_input()` sets handler | ✅ |
| 13 | `test_builder_permission_policy` | `.permissions()` sets policy | ✅ |
| 14 | `test_builder_memory_manager` | `.memory_manager()` sets manager | ✅ |
| 15 | `test_builder_state_store` | `.state_store()` sets store | ✅ |
| 16 | `test_builder_instructions` | `.instructions()` sets text | ✅ |
| 17 | `test_builder_instructions_file` | `.instructions_file()` sets path | ✅ |
| 18 | `test_builder_hook` | `.hook()` registers handler | ✅ |
| 19 | `test_builder_skill` | `.skill()` adds skill | ✅ |
| 20 | `test_builder_subagent` | `.subagent()` adds config | ✅ |
| 21 | `test_builder_mcp_server` | `.mcp_server()` adds MCP | ✅ |
| 22 | `test_builder_connector` | `.connector()` adds connector | ✅ |
| 23 | `test_builder_plugin` | `.plugin()` applies plugin | ✅ |
| 24 | `test_builder_chaining` | Fluent chaining returns self | ✅ |
| 25 | `test_builder_build` | `.build()` returns Agent | ✅ |
| 26 | `test_builder_build_with_model` | `.build()` with model creates agent | ✅ |
| 27 | `test_builder_event_bus` | `.event_bus()` sets bus | ✅ |
| 28 | `test_builder_checkpoint_interval` | `.checkpoint_interval()` sets interval | ✅ |
| 29 | `test_builder_clone` | `clone()` creates independent copy | ✅ |
| 30 | `test_builder_repr` | `__repr__` shows configured keys | ✅ |

---

## 9. Phase 5 — State, Checkpoint & Session Management

**Priority:** High
**Estimated tests:** ~45

### 9.1 `core/state/state.py` — AgentState

**File:** `tests/unit/state/test_agent_state.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_state_creation_defaults` | Default empty state |
| 2 | `test_state_add_messages` | Append messages to state |
| 3 | `test_state_iteration_tracking` | Iteration counter |
| 4 | `test_state_metrics_tracking` | LLM calls, tool calls, tokens |
| 5 | `test_state_cancel_event` | Cancel event set/checked |
| 6 | `test_state_done_flag` | `_done` flag management |
| 7 | `test_state_extensions_set_get` | `set_ext()` and `get_ext()` |
| 8 | `test_state_extensions_not_found` | `get_ext()` returns None |
| 9 | `test_state_transition_history` | Phase transitions recorded |
| 10 | `test_state_metadata` | Arbitrary metadata storage |

### 9.2 `core/state/state_store.py` — StateStore Implementations

**File:** `tests/unit/state/test_state_store.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_inmemory_store_save` | Save state |
| 2 | `test_inmemory_store_load` | Load saved state |
| 3 | `test_inmemory_store_load_nonexistent` | Returns None |
| 4 | `test_inmemory_store_list_runs` | List runs for agent |
| 5 | `test_inmemory_store_delete` | Delete a run |
| 6 | `test_file_store_save_load` | File-based save/load (uses tmp_path) |
| 7 | `test_file_store_list_runs` | List runs from files |
| 8 | `test_file_store_delete` | Delete file-based run |
| 9 | `test_file_store_corrupted_file` | Handle corrupted state file |

### 9.3 `core/state/checkpoint.py` — Checkpoint

**File:** `tests/unit/state/test_checkpoint.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_checkpoint_creation` | Create checkpoint dataclass |
| 2 | `test_checkpoint_serialize` | Serialize to bytes |
| 3 | `test_checkpoint_deserialize` | Deserialize from bytes |
| 4 | `test_checkpoint_roundtrip` | Serialize → deserialize = same |
| 5 | `test_checkpoint_from_state` | Create from AgentState |
| 6 | `test_checkpoint_restore_messages` | Restore Message objects |
| 7 | `test_checkpoint_with_extensions` | Extensions serialized |
| 8 | `test_checkpoint_with_transition_history` | History preserved |
| 9 | `test_checkpoint_large_state` | Large message list handling |
| 10 | `test_checkpoint_corrupted_data` | Graceful error on bad data |

### 9.4 `core/state/session.py` — Session & SessionManager

**File:** `tests/unit/state/test_session.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_session_creation` | Session dataclass |
| 2 | `test_session_touch` | `touch()` updates `updated_at` |
| 3 | `test_session_store_create` | Create new session |
| 4 | `test_session_store_get` | Retrieve session |
| 5 | `test_session_store_list` | List sessions for agent |
| 6 | `test_session_store_add_message` | Add message to session |
| 7 | `test_session_store_get_messages` | Retrieve session messages |
| 8 | `test_session_store_delete` | Delete session |
| 9 | `test_session_manager_create` | Manager delegates to store |
| 10 | `test_session_manager_get` | Manager delegates get |

---

## 10. Phase 6 — Memory System

**Priority:** High
**Estimated tests:** ~80

### 10.1 Memory ABC Contract Tests

**File:** `tests/unit/memory/test_memory_base.py`

These tests run against every Memory implementation via parametrize:

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_memory_add` | Add entry returns ID |
| 2 | `test_memory_search` | Search returns relevant entries |
| 3 | `test_memory_get_context` | Get context string |
| 4 | `test_memory_get` | Get entry by ID |
| 5 | `test_memory_delete` | Delete entry |
| 6 | `test_memory_clear` | Clear all entries |
| 7 | `test_memory_count` | Count entries |
| 8 | `test_memory_empty_search` | Search on empty memory |

### 10.2 Per-Implementation Tests

**Files:** `tests/unit/memory/test_<implementation>.py`

Each implementation has specific tests:

#### ConversationMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_sliding_window` | Window size enforcement |
| 2 | `test_message_ordering` | Chronological order |
| 3 | `test_window_overflow` | Oldest messages dropped |

#### KeyValueMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_kv_set_get` | Set and retrieve by key |
| 2 | `test_kv_update` | Update existing key |
| 3 | `test_kv_delete_key` | Delete specific key |

#### WorkingMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_ephemeral_storage` | Data stored temporarily |
| 2 | `test_scratchpad_operations` | Read/write scratchpad |

#### EpisodicMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_temporal_ordering` | Time-based retrieval |
| 2 | `test_experience_storage` | Store experience records |
| 3 | `test_relevance_scoring` | Relevance-based search |

#### GraphMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_add_entity` | Add entity node |
| 2 | `test_add_relationship` | Add edge between entities |
| 3 | `test_query_relationships` | Query graph structure |
| 4 | `test_entity_context` | Get context for entity |

#### SelfEditingMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_core_memory` | Core memory operations |
| 2 | `test_archival_memory` | Archival storage/retrieval |
| 3 | `test_memory_editing` | Self-edit operations |

#### FileMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_file_persistence` | Write to file, read back |
| 2 | `test_file_not_found` | Handle missing file |
| 3 | `test_concurrent_access` | Concurrent read/write |

#### CompositeMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_composite_add` | Add to all sub-memories |
| 2 | `test_composite_search` | Search across all sub-memories |
| 3 | `test_composite_priority` | Priority ordering |

#### VectorMemory
| # | Test | Validates |
|---|------|-----------|
| 1 | `test_embedding_storage` | Store with embeddings |
| 2 | `test_semantic_search` | Cosine similarity search |
| 3 | `test_mock_embeddings` | Works without real embeddings |

### 10.3 MemoryManager & Strategies

**File:** `tests/unit/memory/test_memory_manager.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_manager_inject_memory` | Calls injection strategy |
| 2 | `test_manager_save_memory` | Calls save strategy |
| 3 | `test_manager_on_iteration_end` | End-of-iteration hook |
| 4 | `test_manager_get_tools` | Returns memory management tools |
| 5 | `test_manager_custom_injection_strategy` | Custom strategy used |
| 6 | `test_manager_custom_save_strategy` | Custom strategy used |
| 7 | `test_manager_custom_query_strategy` | Custom strategy used |
| 8 | `test_manager_startup_shutdown` | Component lifecycle |

### 10.4 Memory Policies

**File:** `tests/unit/memory/test_memory_policies.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_decay_policy` | Time-based decay |
| 2 | `test_importance_policy` | Importance scoring |
| 3 | `test_combined_policies` | Multiple policies applied |

---

## 11. Phase 7 — Events, Hooks & Middleware

**Priority:** High
**Estimated tests:** ~65

### 11.1 `core/events/hooks.py` — HookRegistry & HookContext

**File:** `tests/unit/events/test_hook_registry.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_register_sync_handler` | Register sync handler |
| 2 | `test_register_async_handler` | Register async handler |
| 3 | `test_emit_event` | Emit triggers handlers |
| 4 | `test_emit_no_handlers` | Emit with no handlers (no error) |
| 5 | `test_handler_priority` | Higher priority runs first |
| 6 | `test_multiple_handlers` | All handlers called |
| 7 | `test_remove_handler` | `off()` removes handler |
| 8 | `test_remove_nonexistent_handler` | `off()` with unknown handler |
| 9 | `test_convenience_methods` | `on_agent_run_before()`, etc. |
| 10 | `test_handler_error_isolation` | One handler error doesn't break others |

**File:** `tests/unit/events/test_hook_context.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_hook_context_creation` | Create with event name |
| 2 | `test_hook_context_data` | Data dict access |
| 3 | `test_hook_context_cancel` | `cancel()` sets cancelled flag |
| 4 | `test_hook_context_modify` | `modify()` updates data |
| 5 | `test_hook_context_state_access` | Access to AgentState |

### 11.2 `core/events/event_bus.py` — EventBus

**File:** `tests/unit/events/test_event_bus.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_publish_event` | Publish an AgentEvent |
| 2 | `test_subscribe_handler` | Subscribe to events |
| 3 | `test_subscribe_pattern` | Pattern-based subscription |
| 4 | `test_unsubscribe` | Unsubscribe by ID |
| 5 | `test_publish_triggers_subscriber` | Pub → sub handler called |
| 6 | `test_replay_events` | Replay from timestamp |
| 7 | `test_dead_letter_queue` | Failed events go to DLQ |
| 8 | `test_in_memory_bus` | InMemoryEventBus specific behavior |

### 11.3 Middleware Tests

**File:** `tests/unit/middleware/test_middleware_base.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_middleware_is_abstract` | Cannot instantiate directly |
| 2 | `test_middleware_default_passthrough` | Default methods pass through |
| 3 | `test_pipeline_before_llm` | Pipeline chains `before_llm_call` |
| 4 | `test_pipeline_after_llm` | Pipeline chains `after_llm_call` |
| 5 | `test_pipeline_before_tool` | Pipeline chains `before_tool_call` |
| 6 | `test_pipeline_after_tool` | Pipeline chains `after_tool_call` |
| 7 | `test_pipeline_on_error` | Pipeline chains `on_error` |
| 8 | `test_pipeline_ordering` | Middleware order matters |
| 9 | `test_pipeline_error_suppression` | Middleware can suppress errors |
| 10 | `test_pipeline_stream_chunk` | `on_llm_stream_chunk` filtering |

**File:** `tests/unit/middleware/test_logging_mw.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_logs_llm_call` | LLM call logged |
| 2 | `test_logs_tool_call` | Tool call logged |
| 3 | `test_log_format` | Structured log format |

**File:** `tests/unit/middleware/test_cost_tracker.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_cost_tracking_openai` | OpenAI pricing calculation |
| 2 | `test_cost_tracking_anthropic` | Anthropic pricing calculation |
| 3 | `test_cost_budget_enforcement` | Budget exceeded → raises |
| 4 | `test_cost_tracking_accumulation` | Costs accumulate across calls |
| 5 | `test_cost_unknown_model` | Unknown model falls back to default |
| 6 | `test_cost_reset` | Reset cost tracking |

**File:** `tests/unit/middleware/test_rate_limit.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_rate_limit_under` | Under limit passes through |
| 2 | `test_rate_limit_exceeded` | Over limit blocks/delays |
| 3 | `test_rate_limit_per_user` | Per-user rate limiting |
| 4 | `test_rate_limit_window` | Sliding window behavior |

**File:** `tests/unit/middleware/test_guardrails.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_injection_detection` | Prompt injection detected |
| 2 | `test_safe_content_passes` | Normal content passes through |
| 3 | `test_content_safety_block` | Unsafe content blocked |

**File:** `tests/unit/middleware/test_tracing.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_span_creation` | Tracing spans created |
| 2 | `test_span_attributes` | Correct attributes set |

**File:** `tests/unit/middleware/test_consumers.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_hook_consumer_llm` | Hook-based observability for LLM calls |
| 2 | `test_hook_consumer_tools` | Hook-based observability for tool calls |

**File:** `tests/unit/middleware/test_prometheus.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_metrics_recorded` | Prometheus metrics emitted |
| 2 | `test_counter_incremented` | Call counters work |

---

## 12. Phase 8 — Security & Permissions

**Priority:** High
**Estimated tests:** ~30

### 12.1 `core/security/permissions.py`

**File:** `tests/unit/security/test_permissions.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_permission_result_allow` | `PermissionResult.allow()` |
| 2 | `test_permission_result_deny` | `PermissionResult.deny()` |
| 3 | `test_permission_result_ask` | `PermissionResult.ask()` |
| 4 | `test_allow_all_policy` | AllowAll always allows |
| 5 | `test_ask_always_policy` | AskAlways always asks |
| 6 | `test_allow_reads_ask_writes` | Read allowed, write asks |
| 7 | `test_compound_policy_all_allow` | All sub-policies allow |
| 8 | `test_compound_policy_one_deny` | One deny → overall deny |
| 9 | `test_file_sandbox_policy_allowed` | Path within sandbox |
| 10 | `test_file_sandbox_policy_denied` | Path outside sandbox |
| 11 | `test_file_sandbox_path_traversal` | `../` traversal blocked |
| 12 | `test_network_sandbox_allowed` | URL in allowlist |
| 13 | `test_network_sandbox_denied` | URL not in allowlist |
| 14 | `test_check_file_access` | `check_file_access()` method |
| 15 | `test_check_network_access` | `check_network_access()` method |

### 12.2 `core/security/human_input.py`

**File:** `tests/unit/security/test_human_input.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_human_input_handler_is_abstract` | Cannot instantiate |
| 2 | `test_mock_human_input_approve` | Mock handler approves |
| 3 | `test_mock_human_input_deny` | Mock handler denies |

---

## 13. Phase 9 — Extensions: Skills, Subagents & Plugins

**Priority:** Medium
**Estimated tests:** ~40

### 13.1 `core/extensions/skills.py`

**File:** `tests/unit/extensions/test_skills.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_skill_creation` | Create Skill dataclass |
| 2 | `test_skill_combined_prompt` | `get_combined_prompt()` |
| 3 | `test_skill_from_directory` | Load skill from directory |
| 4 | `test_skill_with_tools` | Skill has associated tools |
| 5 | `test_skill_with_hooks` | Skill has associated hooks |
| 6 | `test_skill_registry_register` | Register skill |
| 7 | `test_skill_registry_get` | Get skill by name |
| 8 | `test_skill_registry_activate` | Activate skills for state |
| 9 | `test_skill_registry_deactivate` | Deactivate skills |
| 10 | `test_skill_registry_get_active_prompts` | Get active skill prompts |

### 13.2 `core/extensions/subagent.py`

**File:** `tests/unit/extensions/test_subagent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_subagent_config_creation` | SubagentConfig defaults |
| 2 | `test_subagent_config_custom` | Custom subagent config |
| 3 | `test_orchestrator_register` | Register subagent config |
| 4 | `test_orchestrator_spawn` | Spawn subagent with task |
| 5 | `test_orchestrator_spawn_background` | Spawn in background |
| 6 | `test_orchestrator_get_result` | Get background result |
| 7 | `test_orchestrator_handoff` | Handoff to another agent |
| 8 | `test_subagent_inherits_memory` | `inherit_memory=True` |
| 9 | `test_subagent_inherits_tools` | `inherit_tools=True` |
| 10 | `test_subagent_inherits_hooks` | `inherit_hooks=True` |

### 13.3 `core/extensions/plugins.py`

**File:** `tests/unit/extensions/test_plugins.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_plugin_is_abstract` | Cannot instantiate |
| 2 | `test_plugin_register` | `register()` modifies builder |
| 3 | `test_apply_plugins_to_builder` | Apply list of plugins |
| 4 | `test_discover_plugins` | Entry point discovery |
| 5 | `test_plugin_name_version` | Plugin metadata |

---

## 14. Phase 10 — MCP & Connectors

**Priority:** Medium
**Estimated tests:** ~35

### 14.1 MCP Module

**File:** `tests/unit/mcp/test_mcp_client.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_mcp_client_creation` | Constructor with URL |
| 2 | `test_mcp_client_connect` | Connect (mocked transport) |
| 3 | `test_mcp_client_disconnect` | Disconnect |
| 4 | `test_mcp_client_list_tools` | List tools from server |
| 5 | `test_mcp_client_call_tool` | Call tool on server |
| 6 | `test_mcp_client_list_resources` | List resources |
| 7 | `test_mcp_client_read_resource` | Read resource |
| 8 | `test_mcp_client_list_prompts` | List prompts |
| 9 | `test_mcp_client_get_prompt` | Get prompt |
| 10 | `test_mcp_client_timeout` | Connection timeout handling |

**File:** `tests/unit/mcp/test_mcp_config.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_config_creation` | MCPServerConfig |
| 2 | `test_config_env_resolution` | Environment variable resolution |

**File:** `tests/unit/mcp/test_mcp_adapter.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_tool_from_mcp` | Convert single MCPTool → Tool |
| 2 | `test_tools_from_mcp` | Convert list of MCPTools |
| 3 | `test_adapter_schema_mapping` | Schema correctly mapped |

**File:** `tests/unit/mcp/test_mcp_bridge.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_bridge_startup` | Connects to MCP server |
| 2 | `test_bridge_shutdown` | Disconnects |
| 3 | `test_bridge_get_tools` | Returns adapted tools |

**File:** `tests/unit/mcp/test_mcp_transport.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_stdio_transport` | Stdio transport |
| 2 | `test_http_transport` | HTTP transport |

### 14.2 Connectors

**File:** `tests/unit/connectors/test_connector.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_connector_is_abstract` | Cannot instantiate |
| 2 | `test_connector_concrete_impl` | Concrete connector works |
| 3 | `test_connector_connect_disconnect` | Lifecycle |
| 4 | `test_connector_get_tools` | Returns tools |
| 5 | `test_connector_health_check` | Health check default |

**File:** `tests/unit/connectors/test_connector_bridge.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_bridge_startup` | Connects all connectors |
| 2 | `test_bridge_shutdown` | Disconnects all |
| 3 | `test_bridge_get_tools` | Aggregates tools from all connectors |

---

## 15. Phase 11 — Workflow: Plan Mode & Structured Output

**Priority:** Medium
**Estimated tests:** ~30

### 15.1 `core/workflow/plan_mode.py`

**File:** `tests/unit/workflow/test_plan_mode.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_plan_step_creation` | PlanStep dataclass |
| 2 | `test_plan_creation` | Plan with steps |
| 3 | `test_plan_state_creation` | PlanState extension |
| 4 | `test_plan_mode_enter_planning` | Enter planning phase |
| 5 | `test_plan_mode_exit_planning` | Exit returns Plan |
| 6 | `test_plan_mode_execute_plan` | Execute plan steps |
| 7 | `test_plan_mode_read_only_tools` | Read-only tools during planning |
| 8 | `test_todo_manager_create` | Create todo item |
| 9 | `test_todo_manager_get` | Get todo by ID |
| 10 | `test_todo_manager_update_status` | Update status |
| 11 | `test_todo_manager_list` | List todos |
| 12 | `test_todo_manager_list_by_status` | Filter by status |
| 13 | `test_todo_status_transitions` | pending → in_progress → completed |

### 15.2 `core/workflow/structured_output.py`

**File:** `tests/unit/workflow/test_structured_output.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_schema_from_pydantic_model` | Convert Pydantic model to JSON schema |
| 2 | `test_schema_from_dict` | Pass through dict schema |
| 3 | `test_parse_structured_simple` | Parse JSON into Pydantic model |
| 4 | `test_parse_structured_nested` | Nested model parsing |
| 5 | `test_parse_structured_list` | Parse list of models |
| 6 | `test_parse_structured_invalid` | Invalid JSON handling |
| 7 | `test_parse_structured_missing_fields` | Missing required fields |

---

## 16. Phase 12 — Persistence Layer

**Priority:** Medium
**Estimated tests:** ~30

### 16.1 `persistence/base.py` — BasePersistence ABC

**File:** `tests/unit/persistence/test_base_persistence.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_base_persistence_is_abstract` | Cannot instantiate |
| 2 | `test_base_persistence_audit_hooks` | Audit hook methods exist |

### 16.2 `persistence/sqlite.py` — SQLitePersistence

**File:** `tests/unit/persistence/test_sqlite_persistence.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_sqlite_init_schema` | Schema created on init |
| 2 | `test_sqlite_save_run` | Save AgentRun |
| 3 | `test_sqlite_get_run` | Retrieve AgentRun |
| 4 | `test_sqlite_get_run_not_found` | Returns None for unknown |
| 5 | `test_sqlite_list_runs` | List runs by agent_id |
| 6 | `test_sqlite_save_event` | Save AgentRunEvent |
| 7 | `test_sqlite_get_events` | Retrieve events for run |
| 8 | `test_sqlite_save_llm_usage` | Save LLM usage data |
| 9 | `test_sqlite_get_llm_usage` | Retrieve LLM usage |
| 10 | `test_sqlite_close` | Connection closed properly |
| 11 | `test_sqlite_concurrent_access` | Thread safety |

### 16.3 `persistence/memory.py` — InMemoryPersistence

**File:** `tests/unit/persistence/test_memory_persistence.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_inmemory_save_get_run` | Save and retrieve |
| 2 | `test_inmemory_list_runs` | List runs |
| 3 | `test_inmemory_save_events` | Save and get events |
| 4 | `test_inmemory_save_usage` | Save and get usage |

---

## 17. Phase 13 — Built-in Tools

**Priority:** Medium
**Estimated tests:** ~25

### 17.1 File Tools

**File:** `tests/unit/built_in_tools/test_file_tools.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_file_read_existing` | Read existing file |
| 2 | `test_file_read_not_found` | File not found handling |
| 3 | `test_file_write_new` | Write new file |
| 4 | `test_file_write_overwrite` | Overwrite existing |
| 5 | `test_file_read_is_tool` | Registered as Tool |
| 6 | `test_file_write_is_tool` | Registered as Tool |

### 17.2 Code Tools

**File:** `tests/unit/built_in_tools/test_code_tools.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_python_execute_simple` | Execute simple expression |
| 2 | `test_python_execute_output` | Capture stdout |
| 3 | `test_python_execute_error` | Handle syntax/runtime errors |
| 4 | `test_shell_execute_simple` | Run shell command |
| 5 | `test_shell_execute_error` | Handle command errors |
| 6 | `test_shell_execute_timeout` | Timeout enforcement |

### 17.3 Web/HTTP Tools

**File:** `tests/unit/built_in_tools/test_web_tools.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_web_fetch_success` | Fetch URL (mocked httpx) |
| 2 | `test_web_fetch_error` | HTTP error handling |
| 3 | `test_web_fetch_timeout` | Timeout handling |

**File:** `tests/unit/built_in_tools/test_http_tools.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_http_get` | GET request |
| 2 | `test_http_post` | POST with body |
| 3 | `test_http_headers` | Custom headers |
| 4 | `test_http_error` | Error handling |

---

## 18. Phase 14 — Context & Credentials

**Priority:** Medium
**Estimated tests:** ~25

### 18.1 `core/context/context.py` — ContextManager

**File:** `tests/unit/context/test_context_manager.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_count_tokens_basic` | Basic token counting |
| 2 | `test_fit_messages_under_limit` | All messages fit |
| 3 | `test_fit_messages_truncate` | Truncate oldest strategy |
| 4 | `test_fit_messages_summarize` | Summarize strategy |
| 5 | `test_reserve_tokens` | Reserve tokens for response |
| 6 | `test_empty_messages` | Handle empty message list |
| 7 | `test_system_message_preserved` | System message never truncated |

### 18.2 `core/context/instructions.py` — InstructionLoader

**File:** `tests/unit/context/test_instruction_loader.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_load_from_string` | Load instructions from string |
| 2 | `test_load_from_file` | Load instructions from file |
| 3 | `test_load_file_not_found` | Handle missing file |
| 4 | `test_template_variables` | Variable substitution |

### 18.3 `credentials/credentials.py` — CredentialResolver

**File:** `tests/unit/credentials/test_credentials.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_env_resolver_found` | Resolve from env var |
| 2 | `test_env_resolver_not_found` | Returns None |
| 3 | `test_env_resolver_with_prefix` | Prefix prepended |
| 4 | `test_env_resolver_optional_default` | `resolve_optional()` with default |
| 5 | `test_vault_resolver_creation` | VaultCredentialResolver init |
| 6 | `test_aws_resolver_creation` | AWSSecretsResolver init |

---

## 19. Phase 15 — CLI

**Priority:** Low
**Estimated tests:** ~10

**File:** `tests/unit/cli/test_cli.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_cli_creation` | AgentCLI constructor |
| 2 | `test_cli_with_agent` | CLI wraps an Agent |
| 3 | `test_cli_command_parsing` | Parse user commands |
| 4 | `test_cli_exit_command` | Handle exit/quit |
| 5 | `test_cli_help_command` | Handle help command |

---

## 20. Phase 16 — Testing Utilities (Meta-Tests)

**Priority:** High (these utilities are used throughout)
**Estimated tests:** ~50

### 20.1 MockLLM

**File:** `tests/unit/testing/test_mock_llm.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_mock_llm_creation` | Empty MockLLM |
| 2 | `test_add_text_response` | Add and retrieve text response |
| 3 | `test_add_tool_call_response` | Add tool call response |
| 4 | `test_add_multiple_responses` | Queue multiple responses |
| 5 | `test_response_ordering` | FIFO ordering |
| 6 | `test_call_count` | Call count tracking |
| 7 | `test_calls_history` | Request history |
| 8 | `test_text_response_helper` | `text_response()` function |
| 9 | `test_tool_call_response_helper` | `tool_call_response()` function |
| 10 | `test_exhausted_responses` | Error when no more responses |

### 20.2 AgentTestHarness

**File:** `tests/unit/testing/test_harness.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_harness_creation` | Create with agent |
| 2 | `test_harness_run_sync` | Synchronous run |
| 3 | `test_harness_arun` | Async run |
| 4 | `test_harness_tool_calls` | Track tool calls made |
| 5 | `test_harness_messages` | Track all messages |
| 6 | `test_harness_with_mock_llm` | Use provided MockLLM |

### 20.3 ToolTestKit

**File:** `tests/unit/testing/test_toolkit.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_toolkit_creation` | ToolTestKit init |
| 2 | `test_toolkit_test_tool` | Test tool execution |
| 3 | `test_toolkit_validate_schema` | Validate tool schema |

### 20.4 Record/Replay

**File:** `tests/unit/testing/test_replay.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_recording_middleware` | Records LLM calls |
| 2 | `test_recording_save` | Save to file |
| 3 | `test_replay_client_from_file` | Load recording |
| 4 | `test_replay_client_call` | Replay matches recording |
| 5 | `test_roundtrip` | Record → save → load → replay |

### 20.5 Other Testing Utilities

**Files:** `tests/unit/testing/test_*.py` (snapshot, benchmark, eval, regression, coverage)

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_snapshot_tester` | Snapshot comparison |
| 2 | `test_benchmark_suite` | Benchmark execution |
| 3 | `test_eval_suite` | Eval suite run |
| 4 | `test_eval_metrics` | exact_match, contains_match, etc. |
| 5 | `test_regression_detector` | Regression detection |
| 6 | `test_coverage_tracker` | Coverage tracking |
| 7 | `test_multi_agent_harness` | MultiAgentTestHarness |

---

## 21. Phase 17 — Integration Tests

**Priority:** High
**Estimated tests:** ~60

These tests validate cross-module interactions using MockLLM (no real API calls).

### 21.1 Agent + Tools

**File:** `tests/integration/test_agent_with_tools.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_agent_calls_single_tool` | LLM requests tool → tool executed → result fed back → final response |
| 2 | `test_agent_calls_multiple_tools` | Sequential tool calls |
| 3 | `test_agent_parallel_tool_calls` | Parallel tool execution |
| 4 | `test_agent_tool_error_recovery` | Tool error → error message → LLM handles gracefully |
| 5 | `test_agent_tool_chain` | Tool A output → Tool B input (multi-step) |
| 6 | `test_agent_no_tool_calls` | Simple text response (no tools needed) |

### 21.2 Agent + Memory

**File:** `tests/integration/test_agent_with_memory.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_memory_injected_into_context` | Memory content added to messages |
| 2 | `test_memory_saved_after_run` | Conversation saved to memory |
| 3 | `test_memory_across_runs` | Second run uses memory from first |
| 4 | `test_memory_tools_available` | Memory management tools work |

### 21.3 Agent + Middleware

**File:** `tests/integration/test_agent_with_middleware.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_logging_middleware_logs_run` | Full run logged |
| 2 | `test_cost_tracker_tracks_run` | Costs tracked across calls |
| 3 | `test_cost_budget_stops_run` | Budget exceeded stops agent |
| 4 | `test_middleware_chain` | Multiple middleware work together |
| 5 | `test_guardrails_block_injection` | Injection attempt blocked |

### 21.4 Agent + Hooks

**File:** `tests/integration/test_agent_with_hooks.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_full_hook_lifecycle` | All hooks fire in order during run |
| 2 | `test_hook_modifies_request` | Hook modifies LLM request |
| 3 | `test_hook_cancels_tool` | Hook prevents tool execution |
| 4 | `test_hook_error_handling` | Hook error doesn't crash agent |

### 21.5 Agent + State/Checkpoint

**File:** `tests/integration/test_agent_with_state.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_state_checkpointed_during_run` | Checkpoint saved after each iteration |
| 2 | `test_resume_from_checkpoint` | Resume run from checkpoint |
| 3 | `test_state_store_persists` | State survives agent restart |

### 21.6 Agent + Sessions

**File:** `tests/integration/test_agent_with_sessions.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_multi_turn_conversation` | Messages persist across turns |
| 2 | `test_session_isolation` | Different sessions don't mix |
| 3 | `test_session_resume` | Resume existing session |

### 21.7 Agent + Permissions

**File:** `tests/integration/test_agent_with_permissions.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_allowed_tool_executes` | Allowed tools run normally |
| 2 | `test_denied_tool_blocked` | Denied tools return permission error |
| 3 | `test_ask_user_flow` | Human-in-the-loop confirmation |

### 21.8 Agent + Skills

**File:** `tests/integration/test_agent_with_skills.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_invoke_skill` | Skill activated and used |
| 2 | `test_skill_tools_available` | Skill tools accessible |
| 3 | `test_skill_prompt_injection` | Skill system prompt added |

### 21.9 Agent + Subagents

**File:** `tests/integration/test_agent_with_subagents.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_spawn_subagent` | Subagent runs and returns result |
| 2 | `test_subagent_tool_inheritance` | Tools inherited when configured |
| 3 | `test_agent_handoff` | Handoff context preserved |

### 21.10 Agent + Plan Mode

**File:** `tests/integration/test_agent_with_plan_mode.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_plan_mode_workflow` | Plan → approve → execute |
| 2 | `test_read_only_tools_during_planning` | Write tools blocked in planning |
| 3 | `test_todo_management` | Todos created/updated during plan |

### 21.11 Agent + Structured Output

**File:** `tests/integration/test_agent_with_structured.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_pydantic_response_format` | Agent returns parsed Pydantic model |
| 2 | `test_dict_response_format` | Agent returns parsed dict |
| 3 | `test_structured_output_validation` | Invalid output handling |

### 21.12 Other Integration Tests

**Files:** Various integration test files

| File | Tests | Validates |
|------|-------|-----------|
| `test_agent_streaming.py` | 3 | Streaming events during agent run |
| `test_agent_builder_full.py` | 3 | Full builder → run pipeline |
| `test_middleware_pipeline.py` | 3 | Multiple middleware stacked |
| `test_memory_persistence.py` | 3 | Memory + persistence backend |
| `test_llm_routing.py` | 3 | Client + Router + Provider chain |
| `test_mcp_integration.py` | 3 | MCP client + bridge + adapter + agent |
| `test_connector_integration.py` | 3 | Connector + bridge + agent |
| `test_plugin_system.py` | 3 | Plugin discovery + registration + agent |
| `test_event_bus_integration.py` | 3 | EventBus + agent lifecycle |
| `test_cost_budget.py` | 3 | Cost tracking + budget enforcement end-to-end |

---

## 22. Phase 18 — End-to-End / Example Tests

**Priority:** Medium
**Estimated tests:** ~25

These tests validate complete agent scenarios as a user would use them, using MockLLM.

### 22.1 Simple Agent

**File:** `tests/e2e/test_simple_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_hello_world_agent` | Minimal agent that responds |
| 2 | `test_agent_with_system_prompt` | Custom system prompt used |
| 3 | `test_agent_run_result_structure` | AgentRunResult has all fields |

### 22.2 Tool Agent

**File:** `tests/e2e/test_tool_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_calculator_agent` | Agent calls calculator tool |
| 2 | `test_multi_tool_agent` | Agent orchestrates multiple tools |
| 3 | `test_tool_error_handling` | Agent handles tool failures |

### 22.3 Multi-Turn Agent

**File:** `tests/e2e/test_multi_turn_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_conversation_continuity` | Context maintained across turns |
| 2 | `test_session_based_conversation` | Session-based multi-turn |

### 22.4 Coding Agent

**File:** `tests/e2e/test_coding_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_file_read_write_agent` | Agent reads and writes files |
| 2 | `test_code_execution_agent` | Agent executes Python code |

### 22.5 Multi-Agent

**File:** `tests/e2e/test_multi_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_parent_child_agents` | Parent spawns child, gets result |
| 2 | `test_agent_handoff` | Agent hands off to specialist |
| 3 | `test_parallel_subagents` | Multiple subagents in parallel |

### 22.6 Resilient Agent

**File:** `tests/e2e/test_resilient_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_agent_timeout_recovery` | Agent handles timeout gracefully |
| 2 | `test_agent_max_iterations` | Stops at max iterations |
| 3 | `test_agent_provider_fallback` | Falls back to another provider |

### 22.7 Memory Agent

**File:** `tests/e2e/test_memory_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_agent_remembers_context` | Memory persists across runs |
| 2 | `test_agent_with_composite_memory` | Multiple memory types |

### 22.8 Structured Agent

**File:** `tests/e2e/test_structured_agent.py`

| # | Test Case | What It Validates |
|---|-----------|-------------------|
| 1 | `test_agent_structured_output` | Returns parsed Pydantic model |
| 2 | `test_agent_json_schema_output` | Returns structured JSON |

---

## 23. Phase 19 — Performance & Stress Tests

**Priority:** Low
**Estimated tests:** ~12

**File:** `tests/performance/test_tool_execution_perf.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_tool_execution_throughput` | 1000 tool executions < 5s |
| 2 | `test_parallel_tool_execution` | Parallel overhead < 2x single |

**File:** `tests/performance/test_memory_operations_perf.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_memory_add_1000` | 1000 entries added < 2s |
| 2 | `test_memory_search_1000` | Search over 1000 entries < 1s |

**File:** `tests/performance/test_state_checkpoint_perf.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_checkpoint_serialize_large` | Serialize 100 messages < 1s |
| 2 | `test_checkpoint_deserialize_large` | Deserialize 100 messages < 1s |

**File:** `tests/performance/test_middleware_overhead.py`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_middleware_pipeline_overhead` | 5 middleware < 10% overhead |
| 2 | `test_hook_emit_overhead` | 10 hooks < 5% overhead |

---

## 24. Phase 20 — CI/CD & Coverage Configuration

### 24.1 GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test]"
      - run: pytest tests/unit -v --cov --cov-report=xml -m "unit"
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[test,structured]"
      - run: pytest tests/integration -v -m "integration"

  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[test,structured]"
      - run: pytest tests/e2e -v -m "e2e"
```

### 24.2 Makefile / Script Commands

Add to project root as `Makefile` or document in README:

```makefile
.PHONY: test test-unit test-integration test-e2e test-perf test-all test-cov

test:                   ## Run all tests (except live and slow)
	pytest tests/ -v -m "not live and not slow"

test-unit:              ## Run unit tests only
	pytest tests/unit -v -m "unit"

test-integration:       ## Run integration tests only
	pytest tests/integration -v -m "integration"

test-e2e:               ## Run end-to-end tests only
	pytest tests/e2e -v -m "e2e"

test-perf:              ## Run performance tests
	pytest tests/performance -v -m "slow"

test-live:              ## Run live API tests (requires API keys)
	pytest tests/live -v -m "live"

test-all:               ## Run everything
	pytest tests/ -v

test-cov:               ## Run with coverage report
	pytest tests/ -v --cov=src/curio_agent_sdk --cov-report=html --cov-report=term -m "not live"

test-fast:              ## Run fast unit tests with parallel execution
	pytest tests/unit -v -x -n auto -m "unit"

test-watch:             ## Run tests on file changes (requires pytest-watch)
	ptw tests/unit -- -v -x
```

---

## 25. Running Tests

### Quick Start

```bash
# Install dev + test dependencies
pip install -e ".[dev,test,structured]"

# Run all unit tests
pytest tests/unit -v

# Run specific test file
pytest tests/unit/tools/test_tool.py -v

# Run specific test
pytest tests/unit/tools/test_tool.py::test_tool_from_function -v

# Run with coverage
pytest tests/unit --cov=src/curio_agent_sdk --cov-report=html

# Run in parallel (faster)
pytest tests/unit -n auto

# Run integration tests
pytest tests/integration -v

# Run everything except live tests
pytest tests/ -v -m "not live"

# Run with verbose failure output
pytest tests/unit -v --tb=short -x  # stop on first failure
```

### Environment Variables for Live Tests

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
pytest tests/live -v -m "live"
```

---

## 26. Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `models/` | 95% | Pure data classes, easy to test fully |
| `core/tools/` | 90% | Critical path, high usage |
| `core/llm/` | 85% | Mocked providers, some edge cases hard to reach |
| `core/agent/` | 85% | Complex orchestration, some async paths |
| `core/state/` | 90% | Serialization must be reliable |
| `core/events/` | 85% | Hook system is critical |
| `core/loops/` | 90% | Core execution logic |
| `core/security/` | 90% | Security must be thoroughly tested |
| `memory/` | 80% | Many implementations, some optional |
| `middleware/` | 80% | Each middleware independently testable |
| `mcp/` | 75% | Heavily mocked, real MCP servers optional |
| `connectors/` | 75% | Abstract layer, minimal logic |
| `persistence/` | 80% | DB backends need schema testing |
| `tools/` (built-in) | 80% | I/O-heavy, some mocking needed |
| `testing/` | 85% | Meta-testing, important for SDK users |
| `cli/` | 60% | Interactive I/O, hard to test fully |
| `resilience/` | 85% | Circuit breaker must be reliable |
| **Overall** | **≥ 85%** | |

---

## 27. Test Conventions & Best Practices

### Naming
- Test files: `test_<module_name>.py`
- Test classes: `TestClassName` (optional, prefer flat functions)
- Test functions: `test_<what_it_does>` — descriptive, action-oriented
- Fixtures: `snake_case`, descriptive names

### Structure (AAA Pattern)
```python
async def test_tool_execute_with_args():
    # Arrange
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Act
    result = await add.execute(a=2, b=3)

    # Assert
    assert result == 5
```

### Async Tests
- All async tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- No need for `@pytest.mark.asyncio` decorator (auto mode handles it)

### Mocking Guidelines
- Use `MockLLM` from `curio_agent_sdk.testing` for LLM mocking
- Use `pytest-mock`'s `mocker` fixture for patching
- Never mock what you own — test real implementations where possible
- Mock at boundaries: HTTP clients, file system, external APIs

### Fixtures Best Practices
- Keep fixtures in `conftest.py` at the appropriate level
- Use `tmp_path` for any file I/O tests
- Prefer factory fixtures over static data
- Use `autouse=True` sparingly

### Parametrize for Variants
```python
@pytest.mark.parametrize("policy_class,expected", [
    (AllowAll, True),
    (AskAlways, None),  # None = asks
])
async def test_permission_policies(policy_class, expected):
    policy = policy_class()
    result = await policy.check_tool_call("any_tool", {})
    assert result.allowed == expected or result.ask_user
```

### Markers
```python
@pytest.mark.unit
async def test_fast_unit_test():
    ...

@pytest.mark.integration
async def test_cross_module():
    ...

@pytest.mark.e2e
async def test_full_workflow():
    ...

@pytest.mark.slow
async def test_performance():
    ...

@pytest.mark.live
async def test_real_api():
    ...
```

### Error Testing
```python
async def test_tool_not_found():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="unknown_tool"):
        registry.get("unknown_tool")
```

### Snapshot/Golden File Testing
For complex outputs (like serialized checkpoints), use snapshot testing:
```python
async def test_checkpoint_serialize_snapshot(snapshot):
    checkpoint = create_test_checkpoint()
    data = checkpoint.serialize()
    assert data == snapshot
```

---

## Summary

| Phase | Focus Area | Est. Tests | Priority |
|-------|-----------|-----------|----------|
| 1 | Models, Exceptions, Base Classes | ✅ 206 (100% cov) | Highest |
| 2 | Tooling System | ✅ 129 (87% cov) | Very High |
| 3 | LLM Client & Providers | ✅ 92 (66% cov) | Very High |
| 4 | Agent Loop & Runtime | ✅ 78 (64% cov) | Very High |
| 5 | State, Checkpoint, Session | ~45 | High |
| 6 | Memory System | ~80 | High |
| 7 | Events, Hooks, Middleware | ~65 | High |
| 8 | Security & Permissions | ~30 | High |
| 9 | Extensions (Skills, Subagents, Plugins) | ~40 | Medium |
| 10 | MCP & Connectors | ~35 | Medium |
| 11 | Plan Mode & Structured Output | ~30 | Medium |
| 12 | Persistence Layer | ~30 | Medium |
| 13 | Built-in Tools | ~25 | Medium |
| 14 | Context & Credentials | ~25 | Medium |
| 15 | CLI | ~10 | Low |
| 16 | Testing Utilities (Meta) | ~50 | High |
| 17 | Integration Tests | ~60 | High |
| 18 | E2E / Example Tests | ~25 | Medium |
| 19 | Performance Tests | ~12 | Low |
| 20 | CI/CD & Coverage | — | Medium |
| **TOTAL** | | **~822** | |

---

## Execution Order Recommendation

1. **Phases 1-4** (Foundation → Tools → LLM → Agent Loop) — These form the core and should be done first. Everything else depends on them.
2. **Phase 16** (Testing Utilities) — Test the testing tools early so they can be used for all subsequent phases.
3. **Phases 5-8** (State, Memory, Events, Security) — Core infrastructure used by integration tests.
4. **Phase 17** (Integration Tests) — Validate cross-module behavior.
5. **Phases 9-15** (Extensions, MCP, Workflow, Persistence, etc.) — Module-specific tests.
6. **Phase 18** (E2E Tests) — Full workflow validation.
7. **Phase 19-20** (Performance, CI/CD) — Final polish.

---

*This plan is a living document. Update it as modules evolve, new features are added, or test gaps are identified.*
