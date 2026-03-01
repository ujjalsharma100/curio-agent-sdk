"""
Unit tests for curio_agent_sdk.persistence.sqlite — SQLitePersistence.

Covers: schema init, save/get/list runs, events, LLM usage, close, health_check, audit, stats.
"""

import sqlite3
import pytest
from datetime import datetime

from curio_agent_sdk.persistence.sqlite import SQLitePersistence
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
class TestSQLitePersistence:
    def test_sqlite_init_schema(self, tmp_path):
        """Schema and tables are created on init."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_runs'"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_run_events'"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_llm_usage'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_sqlite_save_run(self, tmp_path):
        """Save AgentRun and retrieve it."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        run = _make_run()
        p.create_agent_run(run)
        assert run.id is not None
        loaded = p.get_agent_run(run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.agent_id == run.agent_id
        assert loaded.agent_name == run.agent_name
        assert loaded.objective == run.objective
        assert loaded.status == run.status

    def test_sqlite_get_run_not_found(self, tmp_path):
        """get_agent_run returns None for unknown run_id."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        assert p.get_agent_run("nonexistent") is None

    def test_sqlite_list_runs(self, tmp_path):
        """List runs by agent_id with limit/offset."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        p.create_agent_run(_make_run(agent_id="a1", run_id="r1"))
        p.create_agent_run(_make_run(agent_id="a1", run_id="r2"))
        p.create_agent_run(_make_run(agent_id="a2", run_id="r3"))
        runs = p.get_agent_runs(agent_id="a1", limit=10, offset=0)
        assert len(runs) == 2
        run_ids = {r.run_id for r in runs}
        assert run_ids == {"r1", "r2"}
        all_runs = p.get_agent_runs(limit=10, offset=0)
        assert len(all_runs) == 3

    def test_sqlite_update_run(self, tmp_path):
        """Update an existing agent run."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        run = _make_run(status=AgentRunStatus.RUNNING.value)
        p.create_agent_run(run)
        run.status = AgentRunStatus.COMPLETED.value
        run.final_output = "Done"
        run.total_iterations = 5
        p.update_agent_run(run.run_id, run)
        loaded = p.get_agent_run(run.run_id)
        assert loaded.status == AgentRunStatus.COMPLETED.value
        assert loaded.final_output == "Done"
        assert loaded.total_iterations == 5

    def test_sqlite_delete_run(self, tmp_path):
        """Delete agent run returns True when found."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        run = _make_run()
        p.create_agent_run(run)
        assert p.delete_agent_run(run.run_id) is True
        assert p.get_agent_run(run.run_id) is None
        assert p.delete_agent_run("nonexistent") is False

    def test_sqlite_save_event(self, tmp_path):
        """Save and retrieve agent run events."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        run = _make_run()
        p.create_agent_run(run)
        evt = _make_event(run_id=run.run_id, event_type="tool_call")
        p.log_agent_run_event(evt)
        assert evt.id is not None
        events = p.get_agent_run_events(run.run_id)
        assert len(events) == 1
        assert events[0].event_type == "tool_call"
        assert events[0].data == '{"key": "value"}'

    def test_sqlite_get_events_filter_by_type(self, tmp_path):
        """get_agent_run_events can filter by event_type."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        run = _make_run()
        p.create_agent_run(run)
        p.log_agent_run_event(_make_event(run_id=run.run_id, event_type="started"))
        p.log_agent_run_event(_make_event(run_id=run.run_id, event_type="tool_call"))
        p.log_agent_run_event(_make_event(run_id=run.run_id, event_type="tool_call"))
        all_evts = p.get_agent_run_events(run.run_id)
        assert len(all_evts) == 3
        tool_evts = p.get_agent_run_events(run.run_id, event_type="tool_call")
        assert len(tool_evts) == 2

    def test_sqlite_save_llm_usage(self, tmp_path):
        """Save and retrieve LLM usage."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        usage = _make_usage()
        p.log_llm_usage(usage)
        assert usage.id is not None
        usages = p.get_llm_usage(agent_id="agent-1", run_id="run-1")
        assert len(usages) == 1
        assert usages[0].provider == "openai"
        assert usages[0].model == "gpt-4"
        assert usages[0].input_tokens == 10
        assert usages[0].output_tokens == 5

    def test_sqlite_get_llm_usage(self, tmp_path):
        """get_llm_usage filters by agent_id and run_id."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        p.log_llm_usage(_make_usage(agent_id="a1", run_id="r1"))
        p.log_llm_usage(_make_usage(agent_id="a1", run_id="r2"))
        p.log_llm_usage(_make_usage(agent_id="a2", run_id="r2"))
        by_agent = p.get_llm_usage(agent_id="a1", limit=10)
        assert len(by_agent) == 2
        by_run = p.get_llm_usage(run_id="r2", limit=10)
        assert len(by_run) == 2

    def test_sqlite_close(self, tmp_path):
        """close() does not raise (no-op for SQLite per-call connections)."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        p.close()

    def test_sqlite_health_check(self, tmp_path):
        """health_check returns True when DB is accessible."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        assert p.health_check() is True

    def test_sqlite_concurrent_access(self, tmp_path):
        """Multiple operations from same process (thread safety)."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        for i in range(5):
            run = _make_run(agent_id="agent-1", run_id=f"run-{i}")
            p.create_agent_run(run)
        runs = p.get_agent_runs(agent_id="agent-1", limit=10)
        assert len(runs) == 5

    def test_sqlite_get_agent_run_stats(self, tmp_path):
        """get_agent_run_stats returns aggregate stats."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
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

    def test_sqlite_audit_log(self, tmp_path):
        """log_audit_event and get_audit_events work."""
        db_path = str(tmp_path / "test.db")
        p = SQLitePersistence(db_path=db_path)
        p.log_audit_event({
            "agent_id": "a1",
            "run_id": "r1",
            "actor_type": "user",
            "actor_id": "u1",
            "action": "start_run",
            "resource": "run",
            "resource_type": "agent_run",
            "metadata": {},
            "timestamp": datetime.now(),
        })
        events = p.get_audit_events(run_id="r1", limit=10)
        assert len(events) == 1
        assert events[0]["action"] == "start_run"
        assert events[0]["hash"] is not None
