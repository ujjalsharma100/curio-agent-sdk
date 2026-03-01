"""
E2E tests: Simple Agent (Phase 18 §22.1)

Validates minimal agent creation, system prompt usage, and run result structure.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_hello_world_agent():
    """Minimal agent that responds to a greeting."""
    mock = MockLLM()
    mock.add_text_response("Hello! How can I help you today?")

    agent = Agent(llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello!")

    assert result.status == "completed"
    assert result.is_success
    assert "Hello" in result.output
    assert result.total_iterations >= 1
    assert result.total_llm_calls >= 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_with_system_prompt():
    """Custom system prompt is used by the agent."""
    mock = MockLLM()
    mock.add_text_response("Ahoy, matey! How can I help ye?")

    agent = Agent(
        system_prompt="You are a pirate assistant. Always speak like a pirate.",
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello!")

    assert result.status == "completed"
    # Verify the system prompt was sent to the LLM
    assert mock.call_count >= 1
    request = mock.calls[0]
    system_msgs = [m for m in request.messages if m.role == "system"]
    assert len(system_msgs) >= 1
    assert "pirate" in system_msgs[0].content.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_run_result_structure():
    """AgentRunResult has all expected fields populated."""
    mock = MockLLM()
    mock.add_text_response("Here is the result.")

    agent = Agent(system_prompt="Test agent.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Give me a result")

    assert isinstance(result, AgentRunResult)
    assert result.status == "completed"
    assert result.is_success is True
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.total_iterations >= 1
    assert result.total_llm_calls >= 1
    assert result.total_input_tokens >= 0
    assert result.total_output_tokens >= 0
    assert isinstance(result.run_id, str)
    assert len(result.run_id) > 0
    assert result.error is None
    # to_dict should work
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "status" in d
    assert "output" in d
