"""
Unit tests for HookContext — creation, data, cancel, modify, state access.
"""

import pytest

from curio_agent_sdk.core.events.hooks import HookContext


@pytest.mark.unit
class TestHookContext:
    def test_hook_context_creation(self):
        ctx = HookContext(event="agent.run.before")
        assert ctx.event == "agent.run.before"
        assert ctx.data == {}
        assert ctx.state is None
        assert ctx.run_id == ""
        assert ctx.agent_id == ""
        assert ctx.iteration == 0
        assert ctx.cancelled is False

    def test_hook_context_with_data(self):
        ctx = HookContext(event="llm.call.after", data={"request": "req", "response": "res"})
        assert ctx.data["request"] == "req"
        assert ctx.data["response"] == "res"

    def test_hook_context_cancel(self):
        ctx = HookContext(event="tool.call.before")
        assert ctx.cancelled is False
        ctx.cancel()
        assert ctx.cancelled is True

    def test_hook_context_modify(self):
        ctx = HookContext(event="agent.run.before", data={"x": 1})
        ctx.modify("x", 2)
        assert ctx.data["x"] == 2
        ctx.modify("new_key", "value")
        assert ctx.data["new_key"] == "value"

    def test_hook_context_state_access(self):
        state = object()
        ctx = HookContext(event="agent.iteration.after", state=state)
        assert ctx.state is state
