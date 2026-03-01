"""
Integration tests: LLM Routing (Phase 17 §21.12)

Validates Client + Router + Provider chain.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.llm.client import LLMClient
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mock_llm_routing():
    """MockLLM acts as a drop-in for LLMClient routing."""
    mock = MockLLM()
    mock.add_text_response("Routed response.")

    agent = Agent(system_prompt="Test.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Test routing")

    assert result.status == "completed"
    assert mock.call_count == 1
    assert "Routed" in result.output


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_receives_correct_messages():
    """LLM client receives system prompt + user message."""
    mock = MockLLM()
    mock.add_text_response("Got it.")

    agent = Agent(system_prompt="You are a test assistant.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    await harness.run("Hello there")

    assert mock.call_count == 1
    request = mock.calls[0]
    # Should have system prompt and user message
    roles = [m.role for m in request.messages]
    assert "system" in roles
    assert "user" in roles


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_multiple_calls():
    """Multiple LLM calls are routed correctly through the client."""
    mock = MockLLM()
    mock.add_text_response("First response.")
    mock.add_text_response("Second response.")

    agent = Agent(system_prompt="Test.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)

    r1 = await harness.run("First call")
    assert r1.status == "completed"

    r2 = await harness.run("Second call")
    assert r2.status == "completed"
    assert mock.call_count == 2
