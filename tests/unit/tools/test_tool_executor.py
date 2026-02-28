"""
Unit tests for curio_agent_sdk.core.tools.executor

Covers: ToolResult, ToolExecutor execute/execute_all/execute_parallel, caching, hooks
"""

import asyncio
import json
import pytest

from curio_agent_sdk.core.tools.tool import Tool, ToolConfig
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor, ToolResult
from curio_agent_sdk.models.llm import Message, ToolCall


# ---- helpers ----

def _make_tool(name: str, func=None, config=None) -> Tool:
    if func is None:
        def func(x: str = "") -> str:
            f"""Run {name}.

            Args:
                x: Input
            """
            return f"{name}: {x}"

    return Tool(func=func, name=name, config=config)


def _tc(name: str, args: dict | None = None, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args or {})


# ===================================================================
# ToolResult
# ===================================================================


class TestToolResult:
    def test_success(self):
        tr = ToolResult(tool_call_id="c1", tool_name="calc", result="42")
        assert tr.tool_call_id == "c1"
        assert tr.tool_name == "calc"
        assert tr.result == "42"
        assert tr.error is None

    def test_error(self):
        tr = ToolResult(tool_call_id="c1", tool_name="calc", result=None, error="Not found")
        assert tr.error == "Not found"

    def test_is_error_true(self):
        tr = ToolResult(tool_call_id="c1", tool_name="fn", result=None, error="fail")
        assert tr.is_error is True

    def test_is_error_false(self):
        tr = ToolResult(tool_call_id="c1", tool_name="fn", result="ok")
        assert tr.is_error is False

    def test_content_string_result(self):
        tr = ToolResult(tool_call_id="c1", tool_name="fn", result="hello")
        assert tr.content == "hello"

    def test_content_dict_result(self):
        tr = ToolResult(tool_call_id="c1", tool_name="fn", result={"key": "value"})
        content = tr.content
        parsed = json.loads(content)
        assert parsed["key"] == "value"

    def test_content_error(self):
        tr = ToolResult(tool_call_id="c1", tool_name="fn", result=None, error="oops")
        assert "Error:" in tr.content
        assert "oops" in tr.content

    def test_content_non_serializable(self):
        class Custom:
            def __str__(self):
                return "custom_obj"

        tr = ToolResult(tool_call_id="c1", tool_name="fn", result=Custom())
        content = tr.content
        assert isinstance(content, str)

    def test_to_message(self):
        tr = ToolResult(tool_call_id="c1", tool_name="calc", result="42")
        msg = tr.to_message()
        assert isinstance(msg, Message)
        assert msg.role == "tool"
        assert msg.tool_call_id == "c1"
        assert msg.name == "calc"
        assert msg.content == "42"

    def test_to_message_error(self):
        tr = ToolResult(tool_call_id="c1", tool_name="calc", result=None, error="fail")
        msg = tr.to_message()
        assert "Error:" in msg.content


# ===================================================================
# ToolExecutor
# ===================================================================


class TestToolExecutor:
    def _make_executor(self, tools: list[Tool] | None = None) -> ToolExecutor:
        registry = ToolRegistry(tools=tools or [])
        return ToolExecutor(registry=registry)

    @pytest.mark.asyncio
    async def test_execute_success(self):
        def add(a: int, b: int) -> int:
            """Add.

            Args:
                a: First
                b: Second
            """
            return a + b

        t = Tool(func=add, name="add")
        executor = self._make_executor([t])
        result = await executor.execute(_tc("add", {"a": 2, "b": 3}))
        assert result.result == 5
        assert result.is_error is False
        assert result.tool_name == "add"

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        executor = self._make_executor([])
        result = await executor.execute(_tc("missing", {}))
        assert result.is_error is True
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        def bad_tool(x: str) -> str:
            """Bad.

            Args:
                x: Input
            """
            raise ValueError("bad!")

        t = Tool(func=bad_tool, name="bad")
        executor = self._make_executor([t])
        result = await executor.execute(_tc("bad", {"x": "test"}))
        assert result.is_error is True
        assert "bad" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_all_sequential(self):
        call_order = []

        def tool_a(x: str = "") -> str:
            """A.

            Args:
                x: Input
            """
            call_order.append("a")
            return "a"

        def tool_b(x: str = "") -> str:
            """B.

            Args:
                x: Input
            """
            call_order.append("b")
            return "b"

        executor = self._make_executor([
            Tool(func=tool_a, name="a"),
            Tool(func=tool_b, name="b"),
        ])
        results = await executor.execute_all([
            _tc("a", call_id="c1"),
            _tc("b", call_id="c2"),
        ])
        assert len(results) == 2
        assert results[0].result == "a"
        assert results[1].result == "b"
        assert call_order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        async def slow_a(x: str = "") -> str:
            """A.

            Args:
                x: Input
            """
            await asyncio.sleep(0.05)
            return "a"

        async def slow_b(x: str = "") -> str:
            """B.

            Args:
                x: Input
            """
            await asyncio.sleep(0.05)
            return "b"

        executor = self._make_executor([
            Tool(func=slow_a, name="a"),
            Tool(func=slow_b, name="b"),
        ])
        results = await executor.execute_parallel([
            _tc("a", call_id="c1"),
            _tc("b", call_id="c2"),
        ])
        assert len(results) == 2
        assert results[0].result == "a"
        assert results[1].result == "b"

    @pytest.mark.asyncio
    async def test_execute_parallel_partial_failure(self):
        def good(x: str = "") -> str:
            """Good.

            Args:
                x: Input
            """
            return "ok"

        def bad(x: str = "") -> str:
            """Bad.

            Args:
                x: Input
            """
            raise ValueError("fail")

        executor = self._make_executor([
            Tool(func=good, name="good"),
            Tool(func=bad, name="bad"),
        ])
        results = await executor.execute_parallel([
            _tc("good", call_id="c1"),
            _tc("bad", call_id="c2"),
        ])
        assert len(results) == 2
        assert results[0].is_error is False
        assert results[1].is_error is True

    @pytest.mark.asyncio
    async def test_execute_parallel_empty(self):
        executor = self._make_executor([])
        results = await executor.execute_parallel([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_parallel_single(self):
        def fn(x: str = "") -> str:
            """Fn.

            Args:
                x: Input
            """
            return "ok"

        executor = self._make_executor([Tool(func=fn, name="fn")])
        results = await executor.execute_parallel([_tc("fn", call_id="c1")])
        assert len(results) == 1
        assert results[0].result == "ok"

    @pytest.mark.asyncio
    async def test_execute_to_messages(self):
        def fn(x: str = "") -> str:
            """Fn.

            Args:
                x: Input
            """
            return "result"

        executor = self._make_executor([Tool(func=fn, name="fn")])
        messages = await executor.execute_to_messages([_tc("fn", call_id="c1")])
        assert len(messages) == 1
        assert messages[0].role == "tool"
        assert messages[0].content == "result"

    @pytest.mark.asyncio
    async def test_execute_parallel_to_messages(self):
        def fn(x: str = "") -> str:
            """Fn.

            Args:
                x: Input
            """
            return "result"

        executor = self._make_executor([Tool(func=fn, name="fn")])
        messages = await executor.execute_parallel_to_messages([_tc("fn", call_id="c1")])
        assert len(messages) == 1
        assert messages[0].role == "tool"

    @pytest.mark.asyncio
    async def test_caching(self):
        call_count = 0

        def counted(x: str = "") -> str:
            """Counted.

            Args:
                x: Input
            """
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        t = Tool(func=counted, name="counted", config=ToolConfig(cache_ttl=10.0))
        executor = self._make_executor([t])

        r1 = await executor.execute(_tc("counted", {"x": "a"}, "c1"))
        r2 = await executor.execute(_tc("counted", {"x": "a"}, "c2"))
        r3 = await executor.execute(_tc("counted", {"x": "b"}, "c3"))

        assert r1.result == "result_1"
        assert r2.result == "result_1"  # cached
        assert r3.result == "result_2"  # different args
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_idempotency(self):
        call_count = 0

        def idem(x: str = "") -> str:
            """Idem.

            Args:
                x: Input
            """
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        t = Tool(func=idem, name="idem", config=ToolConfig(idempotent=True))
        executor = self._make_executor([t])

        r1 = await executor.execute(_tc("idem", {"x": "a"}, "c1"))
        r2 = await executor.execute(_tc("idem", {"x": "a"}, "c2"))

        assert r1.result == "result_1"
        assert r2.result == "result_1"  # replayed
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_run_id_agent_id_set(self):
        executor = self._make_executor([])
        executor.run_id = "run_1"
        executor.agent_id = "agent_1"
        assert executor.run_id == "run_1"
        assert executor.agent_id == "agent_1"

    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped(self):
        """Unexpected exceptions during tool execution are wrapped."""

        def fn(x: str = "") -> str:
            """Fn.

            Args:
                x: Input
            """
            raise RuntimeError("unexpected")

        executor = self._make_executor([Tool(func=fn, name="fn")])
        result = await executor.execute(_tc("fn", {"x": "a"}))
        assert result.is_error is True
        # The error should be caught and wrapped
        assert "unexpected" in result.error.lower() or "fn" in result.error.lower()
