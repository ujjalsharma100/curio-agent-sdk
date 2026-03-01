"""
Unit tests for curio_agent_sdk.core.agent.agent

Covers: Agent — creation, run/arun/astream, context manager,
invoke_skill, spawn_subagent, builder classmethod, agent_id
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from curio_agent_sdk.core.agent.agent import Agent
from curio_agent_sdk.core.agent.builder import AgentBuilder
from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.models.agent import AgentRunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run_result(output: str = "Done") -> AgentRunResult:
    return AgentRunResult(
        status="completed",
        output=output,
        total_iterations=1,
        total_llm_calls=1,
        total_tool_calls=0,
        total_input_tokens=10,
        total_output_tokens=5,
        run_id="test-run",
    )


# ===================================================================
# Tests
# ===================================================================


class TestAgent:

    def test_agent_creation_minimal(self):
        """Minimal constructor — defaults work."""
        agent = Agent()
        assert agent.agent_name == "Agent"
        assert agent.system_prompt == "You are a helpful assistant."
        assert agent.agent_id.startswith("agent-")
        assert agent.runtime is not None

    def test_agent_creation_full(self):
        """All basic parameters."""
        agent = Agent(
            system_prompt="Custom prompt",
            agent_id="my-agent",
            agent_name="MyAgent",
            max_iterations=10,
            timeout=60.0,
            temperature=0.3,
        )
        assert agent.agent_id == "my-agent"
        assert agent.agent_name == "MyAgent"
        assert agent.system_prompt == "Custom prompt"
        assert agent.max_iterations == 10
        assert agent.timeout == 60.0
        assert agent.temperature == 0.3

    @pytest.mark.asyncio
    async def test_agent_arun(self):
        """arun() delegates to runtime.run()."""
        agent = Agent(agent_id="test")
        mock_result = _mock_run_result("Async result")
        agent.runtime.run = AsyncMock(return_value=mock_result)

        result = await agent.arun("Hello")

        assert result.status == "completed"
        assert result.output == "Async result"
        agent.runtime.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_astream(self):
        """astream() yields events from runtime.stream()."""
        agent = Agent(agent_id="test")

        mock_event = MagicMock()
        mock_event.type = "done"

        async def mock_stream(*args, **kwargs):
            yield mock_event

        agent.runtime.stream = mock_stream

        events = []
        async for event in agent.astream("Hello"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "done"

    @pytest.mark.asyncio
    async def test_agent_context_manager(self):
        """async with lifecycle calls start/close."""
        agent = Agent(agent_id="test")
        agent.runtime.startup_components = AsyncMock()
        agent.runtime.shutdown_components = AsyncMock()

        async with agent as a:
            assert a is agent
            agent.runtime.startup_components.assert_called_once()

        agent.runtime.shutdown_components.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_invoke_skill(self):
        """invoke_skill() raises for unknown skill."""
        agent = Agent(agent_id="test")
        with pytest.raises(ValueError, match="Unknown skill"):
            await agent.invoke_skill("nonexistent", "do something")

    @pytest.mark.asyncio
    async def test_agent_spawn_subagent(self):
        """spawn_subagent() raises without orchestrator."""
        agent = Agent(agent_id="test")
        with pytest.raises(RuntimeError, match="No orchestrator"):
            await agent.spawn_subagent("researcher", "find info")

    def test_agent_builder_classmethod(self):
        """Agent.builder() returns AgentBuilder."""
        builder = Agent.builder()
        assert isinstance(builder, AgentBuilder)

    def test_agent_id_generation(self):
        """Auto-generated agent_id has correct format."""
        agent = Agent()
        assert agent.agent_id.startswith("agent-")
        assert len(agent.agent_id) == len("agent-") + 8  # hex[:8]

    def test_agent_custom_id(self):
        """Custom agent_id preserved."""
        agent = Agent(agent_id="custom-123")
        assert agent.agent_id == "custom-123"

    def test_agent_tools_property(self):
        """tools property returns registered tools."""
        agent = Agent()
        assert isinstance(agent.tools, list)

    def test_agent_repr(self):
        """__repr__ includes id and name."""
        agent = Agent(agent_id="r-agent", agent_name="RepAgent")
        r = repr(agent)
        assert "r-agent" in r
        assert "RepAgent" in r
