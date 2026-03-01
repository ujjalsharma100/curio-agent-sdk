"""
Unit tests for curio_agent_sdk.core.loops.base

Covers: AgentLoop ABC — abstract enforcement, get_output default, stream_step default
"""

import pytest

from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.models.llm import Message


# ---------------------------------------------------------------------------
# Concrete subclass for testing non-abstract methods
# ---------------------------------------------------------------------------

class _ConcreteLoop(AgentLoop):
    """Minimal concrete implementation for testing defaults."""

    async def step(self, state: AgentState) -> AgentState:
        state.iteration += 1
        return state

    def should_continue(self, state: AgentState) -> bool:
        return False


# ===================================================================
# Tests
# ===================================================================


class TestAgentLoop:

    def test_agent_loop_is_abstract(self):
        """Cannot instantiate AgentLoop directly."""
        with pytest.raises(TypeError):
            AgentLoop()

    def test_agent_loop_step_abstract(self):
        """step() must be implemented by subclass."""

        class MissesStep(AgentLoop):
            def should_continue(self, state):
                return False

        with pytest.raises(TypeError):
            MissesStep()

    def test_agent_loop_should_continue_abstract(self):
        """should_continue() must be implemented by subclass."""

        class MissesShouldContinue(AgentLoop):
            async def step(self, state):
                return state

        with pytest.raises(TypeError):
            MissesShouldContinue()

    def test_agent_loop_get_output_default(self):
        """Default get_output() extracts last assistant message."""
        loop = _ConcreteLoop()
        state = AgentState(
            messages=[
                Message.system("sys"),
                Message.user("hello"),
                Message.assistant("world"),
            ]
        )
        assert loop.get_output(state) == "world"

    def test_agent_loop_get_output_empty(self):
        """get_output() returns empty string when no assistant messages."""
        loop = _ConcreteLoop()
        state = AgentState(messages=[Message.user("hello")])
        assert loop.get_output(state) == ""

    @pytest.mark.asyncio
    async def test_agent_loop_stream_step_default(self):
        """Default stream_step falls back to step() and yields events."""
        loop = _ConcreteLoop()
        state = AgentState(
            messages=[
                Message.system("sys"),
                Message.user("hi"),
            ]
        )
        # Add an assistant message so get_output has something
        state.messages.append(Message.assistant("streamed"))

        events = []
        async for event in loop.stream_step(state):
            events.append(event)

        types = [e.type for e in events]
        assert "text_delta" in types
        assert "iteration_end" in types
