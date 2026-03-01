"""
Performance tests: Tool Execution (Phase 19)

Validates tool execution throughput and parallel execution overhead.
"""

import asyncio
import time
import pytest

from curio_agent_sdk.core.tools.tool import tool, Tool
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.models.llm import ToolCall


@tool
def fast_tool(x: str) -> str:
    """A fast tool for throughput testing."""
    return f"result:{x}"


@tool
def cpu_tool(n: int) -> int:
    """A CPU-bound tool."""
    return sum(range(n))


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_execution_throughput():
    """1000 tool executions complete in < 5s."""
    registry = ToolRegistry(tools=[fast_tool])
    executor = ToolExecutor(registry)

    start = time.monotonic()

    for i in range(1000):
        tc = ToolCall(id=f"call_{i}", name="fast_tool", arguments={"x": str(i)})
        result = await executor.execute(tc)
        assert not result.is_error

    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"1000 tool executions took {elapsed:.2f}s (limit: 5s)"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """Parallel tool execution overhead is less than 2x single execution."""
    registry = ToolRegistry(tools=[fast_tool])
    executor = ToolExecutor(registry)

    # Measure single sequential execution of 100 tools
    start_seq = time.monotonic()
    for i in range(100):
        tc = ToolCall(id=f"seq_{i}", name="fast_tool", arguments={"x": str(i)})
        await executor.execute(tc)
    elapsed_seq = time.monotonic() - start_seq

    # Measure parallel execution of 100 tools
    start_par = time.monotonic()
    tool_calls = [
        ToolCall(id=f"par_{i}", name="fast_tool", arguments={"x": str(i)})
        for i in range(100)
    ]
    results = await executor.execute_parallel(tool_calls)
    elapsed_par = time.monotonic() - start_par

    assert len(results) == 100
    for r in results:
        assert not r.is_error

    # Parallel should not be more than 2x sequential (in practice should be faster)
    assert elapsed_par < elapsed_seq * 2.0, (
        f"Parallel {elapsed_par:.3f}s > 2x sequential {elapsed_seq:.3f}s"
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_registry_lookup_performance():
    """Registry lookup for 10000 operations completes quickly."""
    # Register 50 tools
    tools = []
    for i in range(50):
        @tool
        def dynamic_tool(x: str) -> str:
            return x
        dynamic_tool._name = f"tool_{i}"
        dynamic_tool.name = f"tool_{i}"
        t = Tool(func=lambda x="": x, name=f"tool_{i}", description=f"Tool {i}")
        tools.append(t)

    registry = ToolRegistry(tools=tools)

    start = time.monotonic()
    for _ in range(10000):
        t = registry.get("tool_25")
        assert t is not None
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"10000 registry lookups took {elapsed:.2f}s (limit: 2s)"
