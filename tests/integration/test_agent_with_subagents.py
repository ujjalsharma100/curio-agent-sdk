"""
Integration tests: Agent + Subagents (Phase 17 §21.9)

Validates subagent spawning, tool inheritance, and agent handoff.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.extensions.subagent import SubagentConfig
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def parent_tool(x: str) -> str:
    """A tool from the parent agent."""
    return f"Parent: {x}"


@tool
def child_tool(x: str) -> str:
    """A tool specific to the child agent."""
    return f"Child: {x}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spawn_subagent():
    """Subagent runs and returns result to parent."""
    mock = MockLLM()
    # Parent agent's mock isn't used directly for the subagent
    # The subagent will get the parent's LLM by default
    mock.add_text_response("Subagent result: analysis complete.")

    sub_config = SubagentConfig(
        name="analyzer",
        system_prompt="You analyze data.",
        max_iterations=5,
    )

    agent = Agent(
        system_prompt="Parent agent.",
        subagent_configs={"analyzer": sub_config},
        llm=mock,
    )

    result = await agent.spawn_subagent(sub_config, "Analyze this data")

    assert result.status == "completed"
    assert result.output is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subagent_tool_inheritance():
    """Tools are inherited from parent when inherit_tools=True."""
    mock = MockLLM()
    mock.add_tool_call_response("parent_tool", {"x": "inherited"})
    mock.add_text_response("Used parent tool.")

    sub_config = SubagentConfig(
        name="inheritor",
        system_prompt="Use parent tools.",
        inherit_tools=True,
        max_iterations=5,
    )

    agent = Agent(
        system_prompt="Parent.",
        tools=[parent_tool],
        subagent_configs={"inheritor": sub_config},
        llm=mock,
    )

    result = await agent.spawn_subagent(sub_config, "Use the parent tool")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_handoff():
    """Handoff context is preserved when passing to another agent."""
    mock_parent = MockLLM()
    mock_target = MockLLM()
    mock_target.add_text_response("Handled by target agent.")

    parent = Agent(
        agent_id="parent",
        system_prompt="Parent agent.",
        llm=mock_parent,
    )

    target = Agent(
        agent_id="target",
        system_prompt="Target agent that handles handoffs.",
        llm=mock_target,
    )

    result = await parent.handoff(
        target=target,
        context="User needs help with billing.",
    )

    assert result.status == "completed"
    assert result.output is not None
