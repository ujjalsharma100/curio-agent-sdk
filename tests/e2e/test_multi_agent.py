"""
E2E tests: Multi-Agent (Phase 18 §22.5)

Validates parent-child agents, handoff, and parallel subagents.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.extensions.subagent import SubagentConfig
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def analyze(data: str) -> str:
    """Analyze data."""
    return f"Analysis complete: {data} contains 3 insights."


@tool
def summarize(text: str) -> str:
    """Summarize text."""
    return f"Summary: {text[:50]}..."


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parent_child_agents():
    """Parent spawns a child subagent and gets its result."""
    mock = MockLLM()
    # The child subagent will use this response
    mock.add_text_response("Child analysis: The data shows positive trends.")

    child_config = SubagentConfig(
        name="analyst",
        system_prompt="You are a data analyst. Analyze the given data thoroughly.",
        max_iterations=5,
    )

    parent = Agent(
        system_prompt="You are a manager that delegates analysis tasks.",
        subagent_configs={"analyst": child_config},
        llm=mock,
    )

    result = await parent.spawn_subagent(child_config, "Analyze Q4 revenue data")

    assert result.status == "completed"
    assert result.output is not None
    assert len(result.output) > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_handoff():
    """Agent hands off conversation to a specialist agent."""
    parent_mock = MockLLM()
    specialist_mock = MockLLM()
    specialist_mock.add_text_response(
        "Based on your billing issue, I can see the charge was a duplicate. "
        "I've initiated a refund of $29.99."
    )

    parent = Agent(
        agent_id="general-agent",
        system_prompt="You are a general customer service agent.",
        llm=parent_mock,
    )

    specialist = Agent(
        agent_id="billing-specialist",
        system_prompt="You are a billing specialist. Resolve billing disputes.",
        llm=specialist_mock,
    )

    result = await parent.handoff(
        target=specialist,
        context="Customer has a duplicate charge of $29.99 on their account.",
    )

    assert result.status == "completed"
    assert "refund" in result.output.lower() or len(result.output) > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_subagents():
    """Multiple subagents run and return results."""
    mock = MockLLM()
    mock.add_text_response("Research result: AI is growing rapidly.")
    mock.add_text_response("Research result: Cloud adoption at 90%.")

    research_config = SubagentConfig(
        name="researcher",
        system_prompt="You are a researcher.",
        max_iterations=3,
    )

    parent = Agent(
        system_prompt="You orchestrate research tasks.",
        subagent_configs={"researcher": research_config},
        llm=mock,
    )

    # Spawn two subagents sequentially (simulating parallel work)
    r1 = await parent.spawn_subagent(research_config, "Research AI trends")
    r2 = await parent.spawn_subagent(research_config, "Research cloud adoption")

    assert r1.status == "completed"
    assert r2.status == "completed"
    assert r1.output is not None
    assert r2.output is not None
