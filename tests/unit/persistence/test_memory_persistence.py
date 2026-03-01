"""
Unit tests for curio_agent_sdk.persistence.memory — InMemoryPersistence.

Covers: save/get run, list runs, save/get events, save/get usage, clear_all, audit.
"""

import json
import pytest
from datetime import datetime

from curio_agent_sdk.persistence.memory import InMemoryPersistence
from curio_agent_sdk.models.agent import (
    AgentRun,
    AgentRunEvent,
    AgentLLMUsage,
    AgentRunStatus,
)


def _make_run(agent_id="agent-1", run_id="run-1", status=AgentRunStatus.PENDING.value):
    return AgentRun(
        agent_id=agent_id,
        run_id=run_id,
        agent_name="TestAgent",
        objective="Test objective",
        status=status,
        total_iterations=0,
        final_output=None,
        execution_history=None,
    )


def _make_event(agent_id="agent-1", run_id="run-1", event_type="started"):
    return AgentRunEvent(
        agent_id=agent_id,
        run_id=run_id,
        agent_name="TestAgent",
        event_type=event_type,
        data='{"key": "value"}',
    )


def _make_usage(agent_id="agent-1", run_id="run-1"):
    return AgentLLMUsage(
        agent_id=agent_id,
        run_id=run_id,
        provider="openai",
        model="gpt-4",
        input_tokens=10,
        output_tokens=5,
        latency_ms=100,
        status="success",
    )


@pytest.mark.unit
class TestInMemoryPersistence:
    def test_inmemory_save_get_run(self):
        """Save and retrieve agent run."""
        p = InMemoryPersistence()
        run = _make_run()
        p.create_agent_run(run)
        assert run.id is not None
        loaded = p.get_agent_run(run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.agent_id == run.agent_id

    def test_inmemory_list_runs(self):
        """List runs with agent_id filter and pagination."""
        p = InMemoryPersistence()
        p.create_agent_run(_make_run(agent_id="a1", run_id="r1"))
        p.create_agent_run(_make_run(agent_id="a1", run_id="r2"))
        p.create_agent_run(_make_run(agent_id="a2", run_id="r3"))
        runs = p.get_agent_runs(agent_id="a1", limit=10, offset=0)
        assert len(runs) == 2
        runs_all = p.get_agent_runs(limit=2, offset=0)
        assert len(runs_all) == 2
        runs_offset = p.get_agent_runs(limit=10, offset=2)
        assert len(runs_offset) == 1

    def test_inmemory_save_events(self):
        """Save and get events for a run."""
        p = InMemoryPersistence()
        run = _make_run()
        p.create_agent_run(run)
        evt = _make_event(run_id=run.run_id, event_type="tool_call")
        p.log_agent_run_event(evt)
        events = p.get_agent_run_events(run.run_id)
        assert len(events) == 1
        assert events[0].event_type == "tool_call"
        assert events[0].data == '{"key": "value"}'

    def test_inmemory_save_usage(self):
        """Save and get LLM usage."""
        p = InMemoryPersistence()
        usage = _make_usage()
        p.log_llm_usage(usage)
        usages = p.get_llm_usage(agent_id="agent-1", run_id="run-1")
        assert len(usages) == 1
        assert usages[0].provider == "openai"
        assert usages[0].model == "gpt-4"

    def test_inmemory_update_run(self):
        """Update existing run."""
        p = InMemoryPersistence()
        run = _make_run(status=AgentRunStatus.RUNNING.value)
        p.create_agent_run(run)
        run.status = AgentRunStatus.COMPLETED.value
        run.final_output = "Done"
        p.update_agent_run(run.run_id, run)
        loaded = p.get_agent_run(run.run_id)
        assert loaded.status == AgentRunStatus.COMPLETED.value
        assert loaded.final_output == "Done"

    def test_inmemory_delete_run(self):
        """Delete run returns True when found."""
        p = InMemoryPersistence()
        run = _make_run()
        p.create_agent_run(run)
        assert p.delete_agent_run(run.run_id) is True
        assert p.get_agent_run(run.run_id) is None
        assert p.delete_agent_run("nonexistent") is False

    def test_inmemory_get_agent_run_stats(self):
        """get_agent_run_stats returns aggregate stats."""
        p = InMemoryPersistence()
        run1 = _make_run(agent_id="a1", run_id="r1", status=AgentRunStatus.COMPLETED.value)
        run2 = _make_run(agent_id="a1", run_id="r2", status=AgentRunStatus.ERROR.value)
        p.create_agent_run(run1)
        p.create_agent_run(run2)
        p.log_llm_usage(_make_usage(agent_id="a1", run_id="r1"))
        stats = p.get_agent_run_stats(agent_id="a1")
        assert stats["total_runs"] == 2
        assert stats["completed_runs"] == 1
        assert stats["error_runs"] == 1
        assert "total_llm_calls" in stats
        assert "latency_percentiles" in stats

    def test_inmemory_clear_all(self):
        """clear_all wipes all data."""
        p = InMemoryPersistence()
        p.create_agent_run(_make_run())
        p.log_llm_usage(_make_usage())
        p.clear_all()
        assert p.get_agent_runs(limit=10) == []
        assert p.get_llm_usage(limit=10) == []

    def test_inmemory_get_all_data(self):
        """get_all_data returns runs, events, llm_usage for debugging."""
        p = InMemoryPersistence()
        run = _make_run()
        p.create_agent_run(run)
        p.log_agent_run_event(_make_event(run_id=run.run_id))
        p.log_llm_usage(_make_usage())
        data = p.get_all_data()
        assert "runs" in data
        assert "events" in data
        assert "llm_usage" in data
        assert run.run_id in data["runs"]
        assert len(data["llm_usage"]) == 1

    def test_inmemory_audit_events(self):
        """log_audit_event and get_audit_events work."""
        p = InMemoryPersistence()
        p.log_audit_event({"agent_id": "a1", "run_id": "r1", "action": "start"})
        events = p.get_audit_events(run_id="r1", limit=10)
        assert len(events) == 1
        assert events[0]["action"] == "start"
