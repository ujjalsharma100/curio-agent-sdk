"""
Unit tests for ToolTestKit (Phase 16 — Testing Utilities).
"""

import pytest

from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.toolkit import ToolTestKit


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ---------------------------------------------------------------------------
# ToolTestKit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_toolkit_creation():
    """ToolTestKit init."""
    kit = ToolTestKit()
    assert kit.validate_schema is True
    assert kit.calls == []
    kit_no_validate = ToolTestKit(validate_schema=False)
    assert kit_no_validate.validate_schema is False


@pytest.mark.unit
def test_toolkit_test_tool():
    """Test tool execution via mock."""
    registry = ToolRegistry()
    registry.register(add)
    kit = ToolTestKit(validate_schema=True)
    kit._attach_registry(registry)
    kit.mock_tool("add", returns=42)
    # Simulate record_call (as harness would do)
    kit._record_call("add", {"a": 1, "b": 2}, result=42)
    assert len(kit.calls) == 1
    assert kit.calls[0].name == "add"
    assert kit.calls[0].args == {"a": 1, "b": 2}
    assert kit.calls[0].result == 42


@pytest.mark.unit
def test_toolkit_validate_schema():
    """Validate tool schema (required args)."""
    registry = ToolRegistry()
    registry.register(add)
    kit = ToolTestKit(validate_schema=True)
    kit._attach_registry(registry)
    kit._record_call("add", {"a": 10, "b": 20})
    assert len(kit.calls) == 1
    with pytest.raises(Exception):
        kit._record_call("add", {"a": 10})  # missing required 'b'


@pytest.mark.unit
def test_toolkit_assert_tool_called():
    """assert_tool_called passes when matching call exists."""
    kit = ToolTestKit(validate_schema=False)
    kit._record_call("add", {"a": 1, "b": 2})
    kit.assert_tool_called("add", a=1, b=2)
    kit.assert_tool_called("add", a=1)


@pytest.mark.unit
def test_toolkit_assert_tool_not_called():
    """assert_tool_not_called passes when tool was not called."""
    kit = ToolTestKit(validate_schema=False)
    kit._record_call("add", {"a": 1, "b": 2})
    kit.assert_tool_not_called("other")
    with pytest.raises(AssertionError, match="NOT be called"):
        kit.assert_tool_not_called("add")


@pytest.mark.unit
def test_toolkit_assert_call_order():
    """assert_call_order checks ordered subsequence."""
    kit = ToolTestKit(validate_schema=False)
    kit._record_call("a", {})
    kit._record_call("b", {})
    kit._record_call("c", {})
    kit.assert_call_order(["a", "c"])
    kit.assert_call_order(["a", "b", "c"])
    with pytest.raises(AssertionError, match="order"):
        kit.assert_call_order(["c", "a"])


@pytest.mark.unit
def test_toolkit_assert_call_count():
    """assert_call_count checks exact number of calls."""
    kit = ToolTestKit(validate_schema=False)
    kit._record_call("add", {"a": 1, "b": 2})
    kit._record_call("add", {"a": 3, "b": 4})
    kit.assert_call_count("add", 2)
    with pytest.raises(AssertionError, match="2 times"):
        kit.assert_call_count("add", 1)


@pytest.mark.unit
def test_toolkit_get_calls_and_clear():
    """get_calls returns calls for a name; clear_calls clears history."""
    kit = ToolTestKit(validate_schema=False)
    kit._record_call("add", {"a": 1})
    kit._record_call("add", {"a": 2})
    kit._record_call("other", {})
    assert len(kit.get_calls("add")) == 2
    assert len(kit.get_calls("other")) == 1
    kit.clear_calls()
    assert len(kit.calls) == 0
    kit.clear_mocks()
    assert len(kit._mocks) == 0
