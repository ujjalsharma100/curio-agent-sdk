"""
Unit tests for AgentTestHarness (Phase 16 — Testing Utilities).
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


# ---------------------------------------------------------------------------
# Harness creation and runs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_harness_creation():
    """Create with agent."""
    agent = Agent(system_prompt="Test", tools=[])
    harness = AgentTestHarness(agent)
    assert harness.agent is agent
    assert harness.mock_llm is not None
    assert harness.result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_harness_run_sync():
    """Synchronous run completes and returns result."""
    mock = MockLLM()
    mock.add_text_response("Done.")
    agent = Agent(system_prompt="Test", tools=[], llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = harness.run_sync("Hello")
    assert result is not None
    assert result.status == "completed"
    assert "Done" in (result.output or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_harness_arun():
    """Async run completes and returns result."""
    mock = MockLLM()
    mock.add_text_response("Async done.")
    agent = Agent(system_prompt="Test", tools=[], llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hi")
    assert result.status == "completed"
    assert harness.result is result
    assert "Async done" in (result.output or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_harness_tool_calls():
    """Track tool calls made."""
    mock = MockLLM()
    mock.add_tool_call_response("greet", {"name": "Alice"})
    mock.add_text_response("I greeted Alice.")
    agent = Agent(system_prompt="Greet the user.", tools=[greet], llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    await harness.run("Greet Alice")
    assert len(harness.tool_calls) >= 1
    names = [t[0] for t in harness.tool_calls]
    assert "greet" in names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_harness_messages():
    """Track all messages via result."""
    mock = MockLLM()
    mock.add_text_response("Reply.")
    agent = Agent(system_prompt="Test", tools=[], llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello")
    assert result is not None
    assert hasattr(result, "output") or hasattr(result, "messages")


@pytest.mark.unit
def test_harness_with_mock_llm():
    """Use provided MockLLM."""
    mock = MockLLM()
    mock.add_text_response("Custom mock.")
    agent = Agent(system_prompt="Test", tools=[])
    harness = AgentTestHarness(agent, llm=mock)
    assert harness.mock_llm is mock
    result = harness.run_sync("Hi")
    assert mock.call_count == 1
    assert "Custom mock" in (result.output or "")


@pytest.mark.unit
def test_harness_set_llm():
    """set_llm replaces the LLM used by the agent."""
    from curio_agent_sdk.testing.replay import ReplayLLMClient
    mock1 = MockLLM()
    mock1.add_text_response("First")
    agent = Agent(system_prompt="Test", tools=[], llm=mock1)
    harness = AgentTestHarness(agent, llm=mock1)
    mock2 = MockLLM()
    mock2.add_text_response("Replaced")
    harness.set_llm(mock2)
    result = harness.run_sync("Hi")
    assert "Replaced" in (result.output or "")
    assert mock2.call_count == 1
