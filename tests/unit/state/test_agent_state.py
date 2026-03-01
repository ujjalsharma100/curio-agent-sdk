"""
Unit tests for curio_agent_sdk.core.state.state — AgentState.

Covers: default state, messages, iteration, metrics, cancel/done,
extensions, transition history, metadata.
"""

import pytest

from curio_agent_sdk.core.state.state import AgentState, StateExtension
from curio_agent_sdk.models.llm import Message


# ---------------------------------------------------------------------------
# AgentState — creation and defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentStateCreation:
    def test_state_creation_defaults(self):
        """Default empty state."""
        state = AgentState()
        assert state.messages == []
        assert state.tools == []
        assert state.tool_schemas == []
        assert state.iteration == 0
        assert state.max_iterations == 25
        assert state.metadata == {}
        assert state.total_llm_calls == 0
        assert state.total_tool_calls == 0
        assert state.total_input_tokens == 0
        assert state.total_output_tokens == 0
        assert state.current_phase == ""
        assert state.is_cancelled is False
        assert state.is_done is False
        assert state.last_message is None
        assert state.assistant_messages == []

    def test_state_add_messages(self):
        """Append messages to state."""
        state = AgentState()
        state.add_message(Message.user("Hello"))
        state.add_message(Message.assistant("Hi"))
        assert len(state.messages) == 2
        assert state.messages[0].content == "Hello"
        assert state.messages[1].content == "Hi"
        assert state.last_message is not None
        assert state.last_message.content == "Hi"

    def test_state_add_messages_bulk(self):
        """Append multiple messages at once."""
        state = AgentState()
        state.add_messages([
            Message.system("You are helpful."),
            Message.user("Hi"),
        ])
        assert len(state.messages) == 2
        state.add_messages([Message.assistant("Hello!")])
        assert len(state.messages) == 3

    def test_state_iteration_tracking(self):
        """Iteration counter."""
        state = AgentState()
        assert state.iteration == 0
        state.iteration = 1
        assert state.iteration == 1
        state.iteration = 5
        assert state.iteration == 5

    def test_state_metrics_tracking(self):
        """LLM calls, tool calls, tokens."""
        state = AgentState()
        state.record_llm_call(input_tokens=100, output_tokens=50)
        state.record_llm_call(input_tokens=200, output_tokens=80)
        state.record_tool_calls(3)
        assert state.total_llm_calls == 2
        assert state.total_input_tokens == 300
        assert state.total_output_tokens == 130
        assert state.total_tool_calls == 3

    def test_state_cancel_event(self):
        """Cancel event set/checked."""
        state = AgentState()
        assert state.is_cancelled is False
        state.cancel()
        assert state.is_cancelled is True

    def test_state_done_flag(self):
        """_done flag management."""
        state = AgentState()
        assert state.is_done is False
        state.mark_done()
        assert state.is_done is True

    def test_state_extensions_set_get(self):
        """set_ext() and get_ext()."""

        class MyExt:
            pass

        state = AgentState()
        ext = MyExt()
        state.set_ext(ext)
        assert state.get_ext(MyExt) is ext

    def test_state_extensions_not_found(self):
        """get_ext() returns None for missing extension."""

        class MyExt:
            pass

        class OtherExt:
            pass

        state = AgentState()
        state.set_ext(MyExt())
        assert state.get_ext(MyExt) is not None
        assert state.get_ext(OtherExt) is None

    def test_state_transition_history(self):
        """Phase transitions recorded."""
        state = AgentState()
        assert state.get_transition_history() == []
        assert state.current_phase == ""
        state.record_transition("planning")
        state.record_transition("executing")
        history = state.get_transition_history()
        assert len(history) == 2
        assert history[0][0] == "planning"
        assert history[1][0] == "executing"
        assert state.current_phase == "executing"

    def test_state_set_transition_history(self):
        """Restore transition history (e.g. from checkpoint)."""
        state = AgentState()
        state.set_transition_history([("a", 1.0), ("b", 2.0)])
        assert state.current_phase == "b"
        assert state.get_transition_history() == [("a", 1.0), ("b", 2.0)]
        state.set_transition_history([])
        assert state.current_phase == ""

    def test_state_metadata(self):
        """Arbitrary metadata storage."""
        state = AgentState()
        state.metadata["key"] = "value"
        state.metadata["count"] = 42
        assert state.metadata["key"] == "value"
        assert state.metadata["count"] == 42

    def test_state_elapsed_time(self):
        """Elapsed time since creation."""
        state = AgentState()
        assert state.elapsed_time >= 0

    def test_state_assistant_messages(self):
        """assistant_messages filters by role."""
        state = AgentState()
        state.add_message(Message.user("Hi"))
        state.add_message(Message.assistant("Hello"))
        state.add_message(Message.user("Bye"))
        state.add_message(Message.assistant("Goodbye"))
        assert len(state.assistant_messages) == 2
        assert state.assistant_messages[0].content == "Hello"
        assert state.assistant_messages[1].content == "Goodbye"
