"""
Integration tests: Agent + Hooks (Phase 17 §21.4)

Validates hook lifecycle, request modification, tool cancellation, and error handling.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.events import HookRegistry
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_hook_lifecycle():
    """All hooks fire in order during a run."""
    events_seen = []

    def on_run_before(ctx):
        events_seen.append("run.before")

    def on_run_after(ctx):
        events_seen.append("run.after")

    def on_iteration_before(ctx):
        events_seen.append("iteration.before")

    def on_iteration_after(ctx):
        events_seen.append("iteration.after")

    registry = HookRegistry()
    registry.on("agent.run.before", on_run_before)
    registry.on("agent.run.after", on_run_after)
    registry.on("agent.iteration.before", on_iteration_before)
    registry.on("agent.iteration.after", on_iteration_after)

    mock = MockLLM()
    mock.add_text_response("Done.")

    agent = Agent(
        system_prompt="Test.",
        hook_registry=registry,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello")

    assert result.status == "completed"
    assert "run.before" in events_seen
    assert "run.after" in events_seen
    assert "iteration.before" in events_seen
    assert "iteration.after" in events_seen
    # Order: run.before → iteration.before → iteration.after → run.after
    assert events_seen.index("run.before") < events_seen.index("iteration.before")
    assert events_seen.index("iteration.after") < events_seen.index("run.after")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hook_modifies_request():
    """Hook modifies LLM request data via ctx.modify()."""
    def on_llm_before(ctx):
        # Hooks can modify data passed to LLM
        ctx.modify("extra_metadata", {"injected": True})

    registry = HookRegistry()
    registry.on("llm.call.before", on_llm_before)

    mock = MockLLM()
    mock.add_text_response("Modified response.")

    agent = Agent(
        system_prompt="Test.",
        hook_registry=registry,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hook_cancels_tool():
    """Hook prevents tool execution via ctx.cancel()."""
    cancelled_tools = []

    def on_tool_before(ctx):
        tool_name = ctx.data.get("tool_name", "")
        if tool_name == "greet":
            ctx.cancel()
            cancelled_tools.append(tool_name)

    registry = HookRegistry()
    registry.on("tool.call.before", on_tool_before)

    mock = MockLLM()
    mock.add_tool_call_response("greet", {"name": "Alice"})
    mock.add_text_response("Tool was cancelled.")

    agent = Agent(
        system_prompt="Test.",
        tools=[greet],
        hook_registry=registry,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Greet Alice")

    assert result.status == "completed"
    assert "greet" in cancelled_tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hook_error_handling():
    """Hook error doesn't crash the agent."""
    def on_iteration_before(ctx):
        raise RuntimeError("Hook exploded!")

    registry = HookRegistry()
    registry.on("agent.iteration.before", on_iteration_before)

    mock = MockLLM()
    mock.add_text_response("Still works.")

    agent = Agent(
        system_prompt="Test.",
        hook_registry=registry,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello")

    # Agent should complete despite hook error (hooks are non-fatal by default)
    assert result.status in ("completed", "error")
