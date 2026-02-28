"""
Tool-level testing utilities for Curio Agent SDK.

`ToolTestKit` helps you:
- Mock specific tools (return fixed values or raise errors)
- Inspect and assert tool call sequences
- Validate recorded tool arguments against tool schemas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from curio_agent_sdk.core.tools.registry import ToolRegistry


@dataclass
class ToolCallRecord:
    """Record of a single tool call observed during a test."""

    name: str
    args: Dict[str, Any]
    result: Any | None = None
    error: str | None = None


@dataclass
class _ToolMock:
    """Internal representation of a mocked tool."""

    returns: Any | None = None
    side_effect: Callable[[Dict[str, Any]], Any] | Exception | None = None


class ToolTestKit:
    """
    Helper for tool-level testing.

    Typical usage:

        from curio_agent_sdk.testing import ToolTestKit, AgentTestHarness

        kit = ToolTestKit()
        kit.mock_tool("read_file", returns="file content")

        harness = AgentTestHarness(agent, tool_kit=kit)
        ...
        kit.assert_tool_called("read_file", path="test.py")
        kit.assert_call_order(["read_file", "write_file"])
    """

    def __init__(self, validate_schema: bool = True) -> None:
        self.validate_schema = validate_schema
        self._registry: ToolRegistry | None = None
        self._mocks: Dict[str, _ToolMock] = {}
        self._calls: List[ToolCallRecord] = []

    # ------------------------------------------------------------------
    # Wiring / internal API (used by AgentTestHarness)
    # ------------------------------------------------------------------

    def _attach_registry(self, registry: ToolRegistry) -> None:
        """Attach the agent's ToolRegistry for schema validation."""
        self._registry = registry

    def _get_mock(self, name: str) -> Optional[_ToolMock]:
        return self._mocks.get(name)

    def _record_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        """Record a tool call and optionally validate arguments."""
        if self.validate_schema and self._registry is not None:
            try:
                tool = self._registry.get(name)
                # This will raise if required params are missing; we intentionally
                # let the exception surface in tests.
                tool.schema.validate(args)
            except Exception:
                # Let tests see validation failures; do not swallow.
                raise

        self._calls.append(ToolCallRecord(name=name, args=dict(args), result=result, error=error))

    # ------------------------------------------------------------------
    # Public API: mocking
    # ------------------------------------------------------------------

    def mock_tool(
        self,
        name: str,
        *,
        returns: Any | None = None,
        side_effect: Callable[[Dict[str, Any]], Any] | Exception | None = None,
    ) -> None:
        """
        Mock a specific tool by name.

        Args:
            name: Registered tool name to mock.
            returns: Value to return when the tool is called.
            side_effect: Either a callable ``fn(args) -> Any`` that computes
                the result or an Exception instance to raise to simulate errors.
        """
        self._mocks[name] = _ToolMock(returns=returns, side_effect=side_effect)

    def clear_mocks(self) -> None:
        """Remove all configured mocks."""
        self._mocks.clear()

    def clear_calls(self) -> None:
        """Clear recorded call history."""
        self._calls.clear()

    # ------------------------------------------------------------------
    # Public API: inspection
    # ------------------------------------------------------------------

    @property
    def calls(self) -> List[ToolCallRecord]:
        """All recorded tool calls in order."""
        return list(self._calls)

    def get_calls(self, name: str) -> List[ToolCallRecord]:
        """Return all calls for a specific tool name."""
        return [c for c in self._calls if c.name == name]

    # ------------------------------------------------------------------
    # Public API: assertions
    # ------------------------------------------------------------------

    def assert_tool_called(self, name: str, **expected_args: Any) -> None:
        """
        Assert that a tool with *name* was called with at least the given args.

        The assertion passes if any recorded call for *name* has all
        ``expected_args`` key/value pairs.
        """
        for call in self._calls:
            if call.name != name:
                continue
            if all(call.args.get(k) == v for k, v in expected_args.items()):
                return

        details = ", ".join(f"{k}={v!r}" for k, v in expected_args.items())
        raise AssertionError(
            f"Expected tool '{name}' to be called with at least {{{details}}}, "
            f"but no matching call was found. Recorded calls: {self._calls!r}"
        )

    def assert_tool_not_called(self, name: str) -> None:
        """Assert that a tool with *name* was never called."""
        if any(c.name == name for c in self._calls):
            raise AssertionError(f"Expected tool '{name}' to NOT be called, but it was.")

    def assert_call_order(self, expected_order: Sequence[str]) -> None:
        """
        Assert that tools were called in the given order.

        This checks that *expected_order* appears as an ordered subsequence
        of the recorded call names.
        """
        actual = [c.name for c in self._calls]
        if not expected_order:
            return

        idx = 0
        for name in actual:
            if name == expected_order[idx]:
                idx += 1
                if idx == len(expected_order):
                    return

        raise AssertionError(
            f"Expected call order subsequence {list(expected_order)!r}, "
            f"but actual order was {actual!r}"
        )

    def assert_call_count(self, name: str, expected_count: int) -> None:
        """Assert that a tool with *name* was called exactly *expected_count* times."""
        count = sum(1 for c in self._calls if c.name == name)
        if count != expected_count:
            raise AssertionError(
                f"Expected tool '{name}' to be called {expected_count} times, "
                f"but it was called {count} times."
            )

