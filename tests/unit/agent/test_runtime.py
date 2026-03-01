"""
Unit tests for curio_agent_sdk.core.agent.runtime

Covers: Runtime — run, timeout, cancellation, error handling,
memory injection/save, hook lifecycle, checkpoint, create_state,
run_with_state, startup/shutdown, streaming
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from curio_agent_sdk.core.agent.runtime import Runtime
from curio_agent_sdk.core.events import (
    HookRegistry,
    AGENT_RUN_BEFORE,
    AGENT_RUN_AFTER,
    AGENT_RUN_ERROR,
    AGENT_ITERATION_BEFORE,
    AGENT_ITERATION_AFTER,
)
from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.models.llm import Message
from curio_agent_sdk.models.agent import AgentRunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubLoop(AgentLoop):
    """Loop that completes after N steps."""

    def __init__(self, steps: int = 1, output: str = "Done"):
        self._steps = steps
        self._current = 0
        self._output = output

    async def step(self, state: AgentState) -> AgentState:
        self._current += 1
        state.iteration += 1
        state.record_llm_call(10, 5)
        state.add_message(Message.assistant(self._output))
        if self._current >= self._steps:
            state._last_finish_reason = "stop"
        else:
            state._last_finish_reason = "tool_use"
        return state

    def should_continue(self, state: AgentState) -> bool:
        if state.is_cancelled or state.is_done:
            return False
        if state.iteration >= state.max_iterations:
            return False
        return state._last_finish_reason == "tool_use"


class _ErrorLoop(AgentLoop):
    """Loop that raises an error on step."""

    async def step(self, state: AgentState) -> AgentState:
        raise RuntimeError("LLM exploded")

    def should_continue(self, state: AgentState) -> bool:
        return False


def _make_runtime(
    loop: AgentLoop | None = None,
    max_iterations: int = 25,
    timeout: float | None = None,
    memory_manager=None,
    state_store=None,
    hook_registry=None,
    checkpoint_interval: int = 1,
) -> Runtime:
    return Runtime(
        loop=loop or _StubLoop(),
        llm=MagicMock(),
        tool_registry=ToolRegistry(),
        tool_executor=MagicMock(spec=ToolExecutor),
        system_prompt="You are helpful.",
        max_iterations=max_iterations,
        timeout=timeout,
        memory_manager=memory_manager,
        state_store=state_store,
        hook_registry=hook_registry,
        checkpoint_interval=checkpoint_interval,
    )


# ===================================================================
# Tests
# ===================================================================


class TestRuntime:

    @pytest.mark.asyncio
    async def test_runtime_run_simple(self):
        """Single iteration, text response."""
        rt = _make_runtime(loop=_StubLoop(steps=1, output="Hello"))
        result = await rt.run("Hi", agent_id="test")

        assert result.status == "completed"
        assert result.output == "Hello"
        assert result.total_iterations == 1
        assert result.total_llm_calls == 1

    @pytest.mark.asyncio
    async def test_runtime_run_with_tool_call(self):
        """Tool call → tool result → final text (multi-step)."""
        rt = _make_runtime(loop=_StubLoop(steps=2, output="Final"))
        result = await rt.run("Do something", agent_id="test")

        assert result.status == "completed"
        assert result.total_iterations == 2
        assert result.total_llm_calls == 2

    @pytest.mark.asyncio
    async def test_runtime_run_multi_iteration(self):
        """Multiple loop iterations."""
        rt = _make_runtime(loop=_StubLoop(steps=3, output="Multi"))
        result = await rt.run("Complex task", agent_id="test")

        assert result.status == "completed"
        assert result.total_iterations == 3

    @pytest.mark.asyncio
    async def test_runtime_run_max_iterations(self):
        """Hits max iterations limit (loop never finishes)."""
        # Loop always returns tool_use, never finishes
        rt = _make_runtime(loop=_StubLoop(steps=999), max_iterations=3)
        result = await rt.run("Infinite", agent_id="test")

        assert result.status == "completed"
        assert result.total_iterations == 3

    @pytest.mark.asyncio
    async def test_runtime_run_timeout(self):
        """Hits timeout."""

        class _SlowLoop(AgentLoop):
            async def step(self, state):
                await asyncio.sleep(10)
                return state

            def should_continue(self, state):
                return True

        rt = _make_runtime(loop=_SlowLoop(), timeout=0.1)
        result = await rt.run("Slow", agent_id="test")

        assert result.status == "timeout"
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_runtime_run_cancellation(self):
        """Cancel during execution."""

        class _CancelLoop(AgentLoop):
            async def step(self, state):
                state.cancel()
                state.iteration += 1
                state._last_finish_reason = "tool_use"
                return state

            def should_continue(self, state):
                return not state.is_cancelled

        rt = _make_runtime(loop=_CancelLoop())
        result = await rt.run("Cancel me", agent_id="test")

        assert result.status == "completed"
        assert result.total_iterations == 1

    @pytest.mark.asyncio
    async def test_runtime_run_error(self):
        """LLM error during run."""
        rt = _make_runtime(loop=_ErrorLoop())
        result = await rt.run("Error", agent_id="test")

        assert result.status == "error"
        assert "LLM exploded" in result.error

    @pytest.mark.asyncio
    async def test_runtime_memory_injection(self):
        """Memory injected at start."""
        mm = AsyncMock()
        rt = _make_runtime(memory_manager=mm)
        await rt.run("Hello", agent_id="test")

        mm.inject.assert_called_once()
        mm.on_run_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_runtime_memory_save(self):
        """Memory saved after run."""
        mm = AsyncMock()
        rt = _make_runtime(memory_manager=mm)
        await rt.run("Hello", agent_id="test")

        mm.on_run_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_runtime_hook_lifecycle(self):
        """All hooks emitted in order."""
        events = []
        registry = HookRegistry()

        def recorder(ctx):
            events.append(ctx.event)

        registry.on(AGENT_RUN_BEFORE, recorder)
        registry.on(AGENT_RUN_AFTER, recorder)
        registry.on(AGENT_ITERATION_BEFORE, recorder)
        registry.on(AGENT_ITERATION_AFTER, recorder)

        rt = _make_runtime(hook_registry=registry)
        await rt.run("Hook test", agent_id="test")

        assert AGENT_RUN_BEFORE in events
        assert AGENT_ITERATION_BEFORE in events
        assert AGENT_ITERATION_AFTER in events
        assert AGENT_RUN_AFTER in events
        # Order: run_before, iter_before, iter_after, run_after
        assert events.index(AGENT_RUN_BEFORE) < events.index(AGENT_ITERATION_BEFORE)
        assert events.index(AGENT_ITERATION_AFTER) < events.index(AGENT_RUN_AFTER)

    @pytest.mark.asyncio
    async def test_runtime_checkpoint_save(self):
        """State checkpointed after iterations."""
        store = AsyncMock()
        rt = _make_runtime(state_store=store, checkpoint_interval=1)
        await rt.run("Checkpoint test", agent_id="test")

        store.save.assert_called()

    @pytest.mark.asyncio
    async def test_runtime_create_state(self):
        """create_state() helper creates proper state."""
        rt = _make_runtime()
        state = rt.create_state("Hello world")

        assert len(state.messages) == 2
        assert state.messages[0].role == "system"
        assert state.messages[1].role == "user"
        assert "Hello world" in state.messages[1].text

    @pytest.mark.asyncio
    async def test_runtime_create_state_with_context(self):
        """create_state() includes context."""
        rt = _make_runtime()
        state = rt.create_state("Hello", context={"key": "value"})

        assert "key" in state.messages[1].text
        assert "value" in state.messages[1].text

    @pytest.mark.asyncio
    async def test_runtime_run_with_state(self):
        """run_with_state() with pre-built state."""
        rt = _make_runtime(loop=_StubLoop(steps=1, output="Pre-built"))
        state = rt.create_state("Pre-built input")
        result = await rt.run_with_state(state, agent_id="test")

        assert result.status == "completed"
        assert result.output == "Pre-built"

    @pytest.mark.asyncio
    async def test_runtime_startup_shutdown(self):
        """Component lifecycle."""
        rt = _make_runtime()
        # _ensure_components_started sets _components_started = True
        await rt._ensure_components_started()
        assert rt._components_started is True

        await rt.shutdown_components()
        assert rt._components_started is False

    @pytest.mark.asyncio
    async def test_runtime_streaming(self):
        """stream() yields events."""
        rt = _make_runtime(loop=_StubLoop(steps=1, output="Streamed"))
        events = []
        async for event in rt.stream("Stream test", agent_id="test"):
            events.append(event)

        types = [e.type for e in events]
        assert "iteration_start" in types
        assert "done" in types
