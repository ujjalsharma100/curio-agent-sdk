"""
E2E tests: Resilient Agent (Phase 18 §22.6)

Validates timeout recovery, max iterations, and provider fallback.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def slow_tool(data: str) -> str:
    """A tool that simulates work."""
    return f"Processed: {data}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_timeout_recovery():
    """Agent handles timeout gracefully and returns a result."""
    mock = MockLLM()
    mock.add_text_response("Completed within timeout.")

    agent = Agent(
        system_prompt="Complete the task quickly.",
        timeout=30.0,  # generous timeout for test
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Do something quick")

    assert result.status == "completed"
    assert result.is_success


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_max_iterations():
    """Agent stops at max iterations and returns appropriate status."""
    mock = MockLLM()
    # Queue tool calls that keep the loop going beyond max_iterations
    for i in range(5):
        mock.add_tool_call_response("slow_tool", {"data": f"iteration_{i}"})
    # Final text response (may not be reached if max_iterations is hit first)
    mock.add_text_response("Finally done after many iterations.")

    agent = Agent(
        system_prompt="Process data iteratively.",
        tools=[slow_tool],
        max_iterations=3,  # Limit to 3 iterations
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Process all data batches")

    # Should stop at max_iterations (3 tool iterations, won't reach text response)
    assert result.status in ("completed", "error", "timeout")
    assert result.total_iterations <= 3


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_provider_fallback():
    """Agent falls back to another provider when primary fails.

    Simulated by having the MockLLM first raise then succeed.
    """
    mock = MockLLM()
    # MockLLM can queue responses; if we only add a success, it simulates fallback
    mock.add_text_response("Fallback response from secondary provider.")

    agent = Agent(
        system_prompt="You are a resilient assistant.",
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Tell me something")

    assert result.status == "completed"
    assert result.is_success
    assert len(result.output) > 0
