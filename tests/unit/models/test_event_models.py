"""
Unit tests for curio_agent_sdk.models.events

Covers: EventType, AgentEvent, StreamEvent
"""

import pytest
from datetime import datetime

from curio_agent_sdk.models.events import (
    AgentEvent,
    EventType,
    StreamEvent,
)


# ===================================================================
# EventType
# ===================================================================


class TestEventType:
    def test_enum_values_run(self):
        assert EventType.RUN_STARTED.value == "run_started"
        assert EventType.RUN_COMPLETED.value == "run_completed"
        assert EventType.RUN_ERROR.value == "run_error"
        assert EventType.RUN_CANCELLED.value == "run_cancelled"
        assert EventType.RUN_TIMEOUT.value == "run_timeout"

    def test_enum_values_iteration(self):
        assert EventType.ITERATION_STARTED.value == "iteration_started"
        assert EventType.ITERATION_COMPLETED.value == "iteration_completed"

    def test_enum_values_llm(self):
        assert EventType.LLM_CALL_STARTED.value == "llm_call_started"
        assert EventType.LLM_CALL_COMPLETED.value == "llm_call_completed"
        assert EventType.LLM_CALL_ERROR.value == "llm_call_error"
        assert EventType.LLM_CALL_RETRIED.value == "llm_call_retried"

    def test_enum_values_tool(self):
        assert EventType.TOOL_CALL_STARTED.value == "tool_call_started"
        assert EventType.TOOL_CALL_COMPLETED.value == "tool_call_completed"
        assert EventType.TOOL_CALL_ERROR.value == "tool_call_error"
        assert EventType.TOOL_CALL_RETRIED.value == "tool_call_retried"

    def test_enum_values_phases(self):
        assert EventType.PLANNING_STARTED.value == "planning_started"
        assert EventType.PLANNING_COMPLETED.value == "planning_completed"
        assert EventType.CRITIQUE_STARTED.value == "critique_started"
        assert EventType.CRITIQUE_COMPLETED.value == "critique_completed"
        assert EventType.SYNTHESIS_STARTED.value == "synthesis_started"
        assert EventType.SYNTHESIS_COMPLETED.value == "synthesis_completed"

    def test_enum_values_state(self):
        assert EventType.CHECKPOINT_SAVED.value == "checkpoint_saved"
        assert EventType.CHECKPOINT_RESTORED.value == "checkpoint_restored"

    def test_enum_custom(self):
        assert EventType.CUSTOM.value == "custom"

    def test_is_str_enum(self):
        assert isinstance(EventType.RUN_STARTED, str)
        assert EventType.RUN_STARTED == "run_started"

    def test_from_value(self):
        assert EventType("run_completed") == EventType.RUN_COMPLETED

    def test_total_count(self):
        # Ensure we test all members if new ones are added
        assert len(EventType) >= 19


# ===================================================================
# AgentEvent
# ===================================================================


class TestAgentEvent:
    def test_creation(self):
        event = AgentEvent(type=EventType.RUN_STARTED)
        assert event.type == EventType.RUN_STARTED
        assert isinstance(event.timestamp, datetime)
        assert event.data == {}
        assert event.run_id == ""
        assert event.agent_id == ""
        assert event.iteration == 0

    def test_full_creation(self):
        now = datetime.now()
        event = AgentEvent(
            type=EventType.TOOL_CALL_COMPLETED,
            timestamp=now,
            data={"tool": "calc", "result": "42"},
            run_id="run_1",
            agent_id="agent_1",
            iteration=3,
        )
        assert event.type == EventType.TOOL_CALL_COMPLETED
        assert event.timestamp == now
        assert event.data["tool"] == "calc"
        assert event.run_id == "run_1"
        assert event.agent_id == "agent_1"
        assert event.iteration == 3

    def test_to_dict(self):
        event = AgentEvent(
            type=EventType.LLM_CALL_STARTED,
            run_id="r1",
            agent_id="a1",
            iteration=1,
            data={"model": "gpt-4o"},
        )
        d = event.to_dict()
        assert d["type"] == "llm_call_started"
        assert d["run_id"] == "r1"
        assert d["agent_id"] == "a1"
        assert d["iteration"] == 1
        assert d["data"]["model"] == "gpt-4o"
        assert "timestamp" in d

    def test_to_dict_timestamp_format(self):
        event = AgentEvent(type=EventType.RUN_STARTED)
        d = event.to_dict()
        # Should be ISO format string
        datetime.fromisoformat(d["timestamp"])

    def test_default_data_is_independent(self):
        """Each instance should get its own data dict."""
        e1 = AgentEvent(type=EventType.RUN_STARTED)
        e2 = AgentEvent(type=EventType.RUN_STARTED)
        e1.data["key"] = "val"
        assert "key" not in e2.data


# ===================================================================
# StreamEvent
# ===================================================================


class TestStreamEvent:
    def test_text_delta(self):
        event = StreamEvent(type="text_delta", text="Hello")
        assert event.type == "text_delta"
        assert event.text == "Hello"

    def test_tool_call_start(self):
        event = StreamEvent(
            type="tool_call_start",
            tool_name="search",
            tool_args={"q": "test"},
        )
        assert event.type == "tool_call_start"
        assert event.tool_name == "search"
        assert event.tool_args == {"q": "test"}

    def test_tool_call_end(self):
        event = StreamEvent(
            type="tool_call_end",
            tool_name="search",
            tool_result="found 5 results",
        )
        assert event.type == "tool_call_end"
        assert event.tool_result == "found 5 results"

    def test_thinking(self):
        event = StreamEvent(type="thinking", text="Let me think...")
        assert event.type == "thinking"
        assert event.text == "Let me think..."

    def test_iteration_start(self):
        event = StreamEvent(type="iteration_start", iteration=1)
        assert event.type == "iteration_start"
        assert event.iteration == 1

    def test_iteration_end(self):
        event = StreamEvent(type="iteration_end", iteration=2)
        assert event.type == "iteration_end"
        assert event.iteration == 2

    def test_error(self):
        event = StreamEvent(type="error", error="Something failed")
        assert event.type == "error"
        assert event.error == "Something failed"

    def test_done(self):
        event = StreamEvent(type="done")
        assert event.type == "done"

    def test_defaults(self):
        event = StreamEvent(type="text_delta")
        assert event.data is None
        assert event.text is None
        assert event.tool_name is None
        assert event.tool_args is None
        assert event.tool_result is None
        assert event.error is None
        assert event.iteration == 0

    def test_with_data(self):
        event = StreamEvent(type="done", data={"total_tokens": 500})
        assert event.data["total_tokens"] == 500
