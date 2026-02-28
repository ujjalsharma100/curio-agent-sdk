"""
Unit tests for curio_agent_sdk.models.agent

Covers: AgentRunStatus, AgentRun, AgentRunEvent, AgentLLMUsage, AgentRunResult
"""

import json
import pytest
from datetime import datetime

from curio_agent_sdk.models.agent import (
    AgentLLMUsage,
    AgentRun,
    AgentRunEvent,
    AgentRunResult,
    AgentRunStatus,
)


# ===================================================================
# AgentRunStatus
# ===================================================================


class TestAgentRunStatus:
    def test_enum_values(self):
        assert AgentRunStatus.PENDING.value == "pending"
        assert AgentRunStatus.RUNNING.value == "running"
        assert AgentRunStatus.COMPLETED.value == "completed"
        assert AgentRunStatus.ERROR.value == "error"
        assert AgentRunStatus.CANCELLED.value == "cancelled"
        assert AgentRunStatus.TIMEOUT.value == "timeout"

    def test_enum_count(self):
        assert len(AgentRunStatus) == 6

    def test_is_str_enum(self):
        assert isinstance(AgentRunStatus.PENDING, str)
        assert AgentRunStatus.PENDING == "pending"

    def test_enum_from_value(self):
        assert AgentRunStatus("completed") == AgentRunStatus.COMPLETED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AgentRunStatus("invalid")


# ===================================================================
# AgentRunResult
# ===================================================================


class TestAgentRunResult:
    def test_defaults(self):
        result = AgentRunResult(status="completed")
        assert result.status == "completed"
        assert result.output == ""
        assert result.parsed_output is None
        assert result.total_iterations == 0
        assert result.total_llm_calls == 0
        assert result.total_tool_calls == 0
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.run_id == ""
        assert result.error is None
        assert result.messages == []

    def test_completed(self):
        result = AgentRunResult(
            status="completed",
            output="The answer is 42.",
            total_iterations=3,
            total_llm_calls=3,
            total_tool_calls=1,
            total_input_tokens=500,
            total_output_tokens=100,
            run_id="run_abc",
        )
        assert result.output == "The answer is 42."
        assert result.run_id == "run_abc"

    def test_error(self):
        result = AgentRunResult(status="error", error="Something went wrong")
        assert result.status == "error"
        assert result.error == "Something went wrong"

    def test_with_parsed_output(self):
        result = AgentRunResult(
            status="completed",
            output='{"name": "Alice"}',
            parsed_output={"name": "Alice"},
        )
        assert result.parsed_output == {"name": "Alice"}

    def test_metrics(self):
        result = AgentRunResult(
            status="completed",
            total_iterations=5,
            total_llm_calls=10,
            total_tool_calls=3,
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert result.total_iterations == 5
        assert result.total_llm_calls == 10
        assert result.total_tool_calls == 3
        assert result.total_input_tokens == 1000
        assert result.total_output_tokens == 500

    def test_is_success_true(self):
        result = AgentRunResult(status="completed")
        assert result.is_success is True

    def test_is_success_false(self):
        for status in ("error", "cancelled", "timeout"):
            result = AgentRunResult(status=status)
            assert result.is_success is False

    def test_to_dict(self):
        result = AgentRunResult(
            status="completed",
            output="done",
            total_iterations=2,
            total_llm_calls=2,
            total_tool_calls=1,
            total_input_tokens=100,
            total_output_tokens=50,
            run_id="r1",
            error=None,
        )
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["output"] == "done"
        assert d["total_iterations"] == 2
        assert d["run_id"] == "r1"
        assert d["error"] is None

    def test_to_dict_with_error(self):
        result = AgentRunResult(status="error", error="fail")
        d = result.to_dict()
        assert d["error"] == "fail"

    def test_messages_list(self):
        from curio_agent_sdk.models.llm import Message

        msgs = [Message.user("hi"), Message.assistant("hello")]
        result = AgentRunResult(status="completed", messages=msgs)
        assert len(result.messages) == 2


# ===================================================================
# AgentRun
# ===================================================================


class TestAgentRun:
    def test_creation(self):
        run = AgentRun(agent_id="agent_1", run_id="run_1")
        assert run.agent_id == "agent_1"
        assert run.run_id == "run_1"

    def test_defaults(self):
        run = AgentRun(agent_id="a", run_id="r")
        assert run.agent_name == ""
        assert run.objective == ""
        assert run.additional_context is None
        assert run.started_at is None
        assert run.finished_at is None
        assert run.total_iterations == 0
        assert run.total_llm_calls == 0
        assert run.total_tool_calls == 0
        assert run.total_input_tokens == 0
        assert run.total_output_tokens == 0
        assert run.final_output is None
        assert run.execution_history is None
        assert run.status == "pending"
        assert run.error_message is None
        assert run.metadata is None
        assert run.id is None
        assert run.created_at is None
        assert run.updated_at is None

    def test_timing(self):
        now = datetime.now()
        run = AgentRun(
            agent_id="a",
            run_id="r",
            started_at=now,
            finished_at=now,
        )
        assert run.started_at == now
        assert run.finished_at == now

    def test_to_dict(self):
        now = datetime.now()
        run = AgentRun(
            agent_id="a1",
            run_id="r1",
            agent_name="MyAgent",
            objective="Do stuff",
            started_at=now,
            total_iterations=3,
            total_llm_calls=5,
            status="completed",
            final_output="done",
        )
        d = run.to_dict()
        assert d["agent_id"] == "a1"
        assert d["run_id"] == "r1"
        assert d["agent_name"] == "MyAgent"
        assert d["total_iterations"] == 3
        assert d["status"] == "completed"
        assert d["started_at"] == now.isoformat()

    def test_to_dict_none_times(self):
        run = AgentRun(agent_id="a", run_id="r")
        d = run.to_dict()
        assert d["started_at"] is None
        assert d["finished_at"] is None

    def test_from_dict(self):
        now = datetime.now()
        data = {
            "agent_id": "a1",
            "run_id": "r1",
            "agent_name": "Test",
            "objective": "obj",
            "started_at": now.isoformat(),
            "finished_at": None,
            "total_iterations": 2,
            "total_llm_calls": 3,
            "total_tool_calls": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "status": "completed",
        }
        run = AgentRun.from_dict(data)
        assert run.agent_id == "a1"
        assert run.run_id == "r1"
        assert run.agent_name == "Test"
        assert run.total_iterations == 2
        assert run.status == "completed"
        assert run.started_at is not None

    def test_from_dict_minimal(self):
        data = {"agent_id": "a", "run_id": "r"}
        run = AgentRun.from_dict(data)
        assert run.agent_id == "a"
        assert run.agent_name == ""
        assert run.status == "pending"

    def test_roundtrip(self):
        now = datetime.now()
        original = AgentRun(
            agent_id="a1",
            run_id="r1",
            agent_name="Agent",
            started_at=now,
            total_iterations=5,
            status="completed",
        )
        d = original.to_dict()
        restored = AgentRun.from_dict(d)
        assert restored.agent_id == original.agent_id
        assert restored.run_id == original.run_id
        assert restored.total_iterations == original.total_iterations
        assert restored.status == original.status


# ===================================================================
# AgentRunEvent
# ===================================================================


class TestAgentRunEvent:
    def test_creation(self):
        event = AgentRunEvent(agent_id="a", run_id="r")
        assert event.agent_id == "a"
        assert event.run_id == "r"

    def test_defaults(self):
        event = AgentRunEvent(agent_id="a", run_id="r")
        assert event.agent_name == ""
        assert event.timestamp is None
        assert event.event_type == ""
        assert event.data is None
        assert event.id is None
        assert event.created_at is None

    def test_to_dict(self):
        now = datetime.now()
        event = AgentRunEvent(
            agent_id="a",
            run_id="r",
            agent_name="Agent",
            timestamp=now,
            event_type="tool_call",
            data='{"tool": "calc"}',
        )
        d = event.to_dict()
        assert d["agent_id"] == "a"
        assert d["event_type"] == "tool_call"
        assert d["timestamp"] == now.isoformat()

    def test_from_dict(self):
        now = datetime.now()
        data = {
            "agent_id": "a",
            "run_id": "r",
            "agent_name": "Agent",
            "timestamp": now.isoformat(),
            "event_type": "llm_call",
            "data": '{"model": "gpt-4o"}',
        }
        event = AgentRunEvent.from_dict(data)
        assert event.agent_id == "a"
        assert event.event_type == "llm_call"
        assert event.timestamp is not None

    def test_get_data_dict_valid_json(self):
        event = AgentRunEvent(
            agent_id="a",
            run_id="r",
            data='{"key": "value", "num": 42}',
        )
        d = event.get_data_dict()
        assert d["key"] == "value"
        assert d["num"] == 42

    def test_get_data_dict_invalid_json(self):
        event = AgentRunEvent(agent_id="a", run_id="r", data="not json")
        d = event.get_data_dict()
        assert d == {"raw": "not json"}

    def test_get_data_dict_none(self):
        event = AgentRunEvent(agent_id="a", run_id="r", data=None)
        d = event.get_data_dict()
        assert d == {}

    def test_get_data_dict_empty_string(self):
        event = AgentRunEvent(agent_id="a", run_id="r", data="")
        d = event.get_data_dict()
        assert d == {}


# ===================================================================
# AgentLLMUsage
# ===================================================================


class TestAgentLLMUsage:
    def test_defaults(self):
        usage = AgentLLMUsage()
        assert usage.agent_id is None
        assert usage.run_id is None
        assert usage.provider == ""
        assert usage.model == ""
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.latency_ms == 0
        assert usage.status == "success"
        assert usage.error_message is None

    def test_full_creation(self):
        usage = AgentLLMUsage(
            agent_id="a1",
            run_id="r1",
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
            latency_ms=350,
            status="success",
        )
        assert usage.provider == "openai"
        assert usage.model == "gpt-4o"

    def test_total_tokens(self):
        usage = AgentLLMUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_to_dict(self):
        usage = AgentLLMUsage(
            agent_id="a",
            run_id="r",
            provider="anthropic",
            model="claude-sonnet",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
            status="success",
        )
        d = usage.to_dict()
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-sonnet"
        assert d["input_tokens"] == 200
        assert d["output_tokens"] == 100
        assert d["latency_ms"] == 500

    def test_error_usage(self):
        usage = AgentLLMUsage(
            status="error",
            error_message="Timeout",
        )
        assert usage.status == "error"
        assert usage.error_message == "Timeout"
