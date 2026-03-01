"""
Integration tests: Agent Builder Full Pipeline (Phase 17 §21.12)

Validates full builder → run pipeline.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.agent.builder import AgentBuilder
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def echo(text: str) -> str:
    """Echo back the input."""
    return text


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_builder_to_run():
    """Build agent via builder and run it successfully."""
    mock = MockLLM()
    mock.add_text_response("Built and running.")

    agent = (
        Agent.builder()
        .system_prompt("Built via builder.")
        .agent_name("BuiltAgent")
        .max_iterations(5)
        .llm(mock)
        .build()
    )

    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Test builder")

    assert result.status == "completed"
    assert agent.agent_name == "BuiltAgent"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_builder_with_tools_and_middleware():
    """Builder with tools and middleware produces a working agent."""
    mock = MockLLM()
    mock.add_tool_call_response("echo", {"text": "hello"})
    mock.add_text_response("Echoed: hello")

    agent = (
        Agent.builder()
        .system_prompt("Echo agent.")
        .add_tool(echo)
        .add_middleware(LoggingMiddleware())
        .llm(mock)
        .build()
    )

    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Echo hello")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_builder_clone_produces_independent_agent():
    """Cloned builder produces an independent agent."""
    mock1 = MockLLM()
    mock1.add_text_response("Agent 1.")
    mock2 = MockLLM()
    mock2.add_text_response("Agent 2.")

    builder = Agent.builder().system_prompt("Base prompt.").max_iterations(3)

    agent1 = builder.clone().agent_name("Agent1").llm(mock1).build()
    agent2 = builder.clone().agent_name("Agent2").llm(mock2).build()

    harness1 = AgentTestHarness(agent1, llm=mock1)
    harness2 = AgentTestHarness(agent2, llm=mock2)

    r1 = await harness1.run("Test 1")
    r2 = await harness2.run("Test 2")

    assert r1.status == "completed"
    assert r2.status == "completed"
    assert agent1.agent_name == "Agent1"
    assert agent2.agent_name == "Agent2"
