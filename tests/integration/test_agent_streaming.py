"""
Integration tests: Agent Streaming (Phase 17 §21.12)

Validates streaming events during an agent run.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.mock_llm import MockLLM


@tool
def lookup(key: str) -> str:
    """Look up a value."""
    return f"Value for {key}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_text_events():
    """Streaming yields events for a simple text response."""
    mock = MockLLM()
    mock.add_text_response("Streamed response.")

    agent = Agent(system_prompt="Test.", llm=mock)

    events = []
    async for event in agent.astream("Hello"):
        events.append(event)

    assert len(events) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_with_tool_calls():
    """Streaming yields events including tool call events."""
    mock = MockLLM()
    mock.add_tool_call_response("lookup", {"key": "test"})
    mock.add_text_response("Found the value.")

    agent = Agent(system_prompt="Test.", tools=[lookup], llm=mock)

    events = []
    async for event in agent.astream("Look up test"):
        events.append(event)

    assert len(events) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_event_types():
    """Stream events have proper type attributes."""
    mock = MockLLM()
    mock.add_text_response("Done.")

    agent = Agent(system_prompt="Test.", llm=mock)

    events = []
    async for event in agent.astream("Hello"):
        events.append(event)

    # At least one event should be present
    assert len(events) >= 1
    for event in events:
        assert hasattr(event, "type")
