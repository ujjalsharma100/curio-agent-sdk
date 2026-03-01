"""
Unit tests for curio_agent_sdk.core.loops.tool_calling

Covers: ToolCallingLoop — step (text/tool), should_continue logic,
parallel vs sequential tool calls, error handling, context manager
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from curio_agent_sdk.core.loops.tool_calling import ToolCallingLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolSchema,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(
    parallel_tool_calls: bool = True,
    response_format=None,
) -> ToolCallingLoop:
    """Create a ToolCallingLoop with mocked LLM and executor."""
    llm = AsyncMock()
    executor = AsyncMock(spec=ToolExecutor)
    loop = ToolCallingLoop(
        llm=llm,
        tool_executor=executor,
        tier="tier1",
        temperature=0.5,
        max_tokens=2048,
        parallel_tool_calls=parallel_tool_calls,
        response_format=response_format,
    )
    return loop


def _text_response(content: str = "Hello!") -> LLMResponse:
    return LLMResponse(
        message=Message.assistant(content),
        provider="openai",
        model="gpt-4o-mini",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _tool_response(tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    tcs = tool_calls or [ToolCall(id="tc1", name="search", arguments={"q": "test"})]
    return LLMResponse(
        message=Message.assistant("", tool_calls=tcs),
        provider="openai",
        model="gpt-4o-mini",
        usage=TokenUsage(input_tokens=15, output_tokens=10),
        finish_reason="tool_use",
    )


def _make_state(tools: bool = False) -> AgentState:
    state = AgentState(
        messages=[
            Message.system("You are helpful."),
            Message.user("Hi"),
        ],
        max_iterations=10,
    )
    if tools:
        state.tool_schemas = [
            ToolSchema(name="search", description="Search", parameters={"type": "object", "properties": {}}),
        ]
        state.tools = [MagicMock()]
    return state


# ===================================================================
# Tests
# ===================================================================


class TestToolCallingLoop:

    @pytest.mark.asyncio
    async def test_loop_step_text_response(self):
        """LLM returns text → state updated, iteration incremented."""
        loop = _make_loop()
        loop.llm.call = AsyncMock(return_value=_text_response("World"))
        state = _make_state()

        state = await loop.step(state)

        assert state.iteration == 1
        assert state.total_llm_calls == 1
        assert state._last_finish_reason == "stop"
        # Assistant message added
        assert any(m.role == "assistant" for m in state.messages)

    @pytest.mark.asyncio
    async def test_loop_step_tool_call(self):
        """LLM returns tool call → tool executed, result added."""
        loop = _make_loop()
        loop.llm.call = AsyncMock(return_value=_tool_response())

        result_msg = Message.tool_result(tool_call_id="tc1", content="result data")
        # Single tool call uses execute_to_messages (parallel requires > 1 calls)
        loop.tool_executor.execute_to_messages = AsyncMock(return_value=[result_msg])

        state = _make_state(tools=True)
        state = await loop.step(state)

        assert state.iteration == 1
        assert state.total_tool_calls == 1
        assert state._last_finish_reason == "tool_use"
        loop.tool_executor.execute_to_messages.assert_called_once()

    @pytest.mark.asyncio
    async def test_loop_step_multiple_tool_calls(self):
        """Multiple tool calls in one step."""
        loop = _make_loop()
        tcs = [
            ToolCall(id="tc1", name="search", arguments={"q": "a"}),
            ToolCall(id="tc2", name="fetch", arguments={"url": "b"}),
        ]
        loop.llm.call = AsyncMock(return_value=_tool_response(tcs))
        loop.tool_executor.execute_parallel_to_messages = AsyncMock(return_value=[
            Message.tool_result(tool_call_id="tc1", content="r1"),
            Message.tool_result(tool_call_id="tc2", content="r2"),
        ])

        state = _make_state(tools=True)
        state = await loop.step(state)

        assert state.total_tool_calls == 2

    @pytest.mark.asyncio
    async def test_loop_step_parallel_tool_calls(self):
        """parallel_tool_calls=True uses execute_parallel_to_messages."""
        loop = _make_loop(parallel_tool_calls=True)
        tcs = [
            ToolCall(id="tc1", name="a", arguments={}),
            ToolCall(id="tc2", name="b", arguments={}),
        ]
        loop.llm.call = AsyncMock(return_value=_tool_response(tcs))
        loop.tool_executor.execute_parallel_to_messages = AsyncMock(return_value=[])

        state = _make_state(tools=True)
        await loop.step(state)

        loop.tool_executor.execute_parallel_to_messages.assert_called_once()
        loop.tool_executor.execute_to_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_step_sequential_tool_calls(self):
        """parallel_tool_calls=False uses execute_to_messages."""
        loop = _make_loop(parallel_tool_calls=False)
        tcs = [
            ToolCall(id="tc1", name="a", arguments={}),
            ToolCall(id="tc2", name="b", arguments={}),
        ]
        loop.llm.call = AsyncMock(return_value=_tool_response(tcs))
        loop.tool_executor.execute_to_messages = AsyncMock(return_value=[])

        state = _make_state(tools=True)
        await loop.step(state)

        loop.tool_executor.execute_to_messages.assert_called_once()
        loop.tool_executor.execute_parallel_to_messages.assert_not_called()

    def test_loop_should_continue_stop(self):
        """finish_reason='stop' → should_continue returns False."""
        loop = _make_loop()
        state = _make_state()
        state._last_finish_reason = "stop"
        assert loop.should_continue(state) is False

    def test_loop_should_continue_tool_use(self):
        """finish_reason='tool_use' → should_continue returns True."""
        loop = _make_loop()
        state = _make_state()
        state._last_finish_reason = "tool_use"
        assert loop.should_continue(state) is True

    def test_loop_should_continue_max_iterations(self):
        """At max iterations → should_continue returns False."""
        loop = _make_loop()
        state = _make_state()
        state._last_finish_reason = "tool_use"
        state.iteration = 10
        state.max_iterations = 10
        assert loop.should_continue(state) is False

    def test_loop_should_continue_cancelled(self):
        """Cancelled state → should_continue returns False."""
        loop = _make_loop()
        state = _make_state()
        state._last_finish_reason = "tool_use"
        state.cancel()
        assert loop.should_continue(state) is False

    @pytest.mark.asyncio
    async def test_loop_context_manager_integration(self):
        """Context manager's fit_messages is called when set."""
        loop = _make_loop()
        loop.llm.call = AsyncMock(return_value=_text_response())

        ctx_mgr = MagicMock()
        ctx_mgr.fit_messages = MagicMock(side_effect=lambda msgs, **kw: msgs)
        loop.context_manager = ctx_mgr

        state = _make_state()
        await loop.step(state)

        ctx_mgr.fit_messages.assert_called_once()

    @pytest.mark.asyncio
    async def test_loop_with_response_format(self):
        """response_format is passed when no tools present."""
        loop = _make_loop(response_format={"type": "json_object"})
        loop.llm.call = AsyncMock(return_value=_text_response('{"key": "value"}'))

        state = _make_state(tools=False)
        await loop.step(state)

        call_args = loop.llm.call.call_args
        request = call_args[0][0]
        # response_format should be set on the request when no tools
        assert request.response_format is not None

    @pytest.mark.asyncio
    async def test_loop_error_in_llm_call(self):
        """LLM error propagates."""
        loop = _make_loop()
        loop.llm.call = AsyncMock(side_effect=RuntimeError("LLM down"))

        state = _make_state()
        with pytest.raises(RuntimeError, match="LLM down"):
            await loop.step(state)

    @pytest.mark.asyncio
    async def test_loop_error_in_tool_execution(self):
        """Tool error propagates from executor."""
        loop = _make_loop()
        # Use multiple tool calls to trigger parallel path
        tcs = [
            ToolCall(id="tc1", name="a", arguments={}),
            ToolCall(id="tc2", name="b", arguments={}),
        ]
        loop.llm.call = AsyncMock(return_value=_tool_response(tcs))
        loop.tool_executor.execute_parallel_to_messages = AsyncMock(
            side_effect=RuntimeError("Tool failed")
        )

        state = _make_state(tools=True)
        with pytest.raises(RuntimeError, match="Tool failed"):
            await loop.step(state)

    @pytest.mark.asyncio
    async def test_loop_no_llm_raises(self):
        """Step without LLM raises RuntimeError."""
        loop = ToolCallingLoop()
        state = _make_state()
        with pytest.raises(RuntimeError, match="LLMClient not set"):
            await loop.step(state)
