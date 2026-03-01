"""
Unit tests for AgentCoverageTracker and merge_coverage_reports (Phase 16).
"""

import pytest

from curio_agent_sdk.core.events import HookRegistry, HookContext, TOOL_CALL_AFTER, TOOL_CALL_ERROR
from curio_agent_sdk.testing.coverage import AgentCoverageTracker, merge_coverage_reports


@pytest.mark.unit
def test_coverage_tracker():
    """Coverage tracking: tools called, hooks emitted."""
    registry = HookRegistry()
    tracker = AgentCoverageTracker()
    tracker.register(registry)
    ctx = HookContext(event=TOOL_CALL_AFTER, data={"tool_name": "read_file"}, agent_id="a", run_id="r")
    tracker._on_event(ctx)
    ctx2 = HookContext(event=TOOL_CALL_AFTER, data={"tool_name": "write_file"}, agent_id="a", run_id="r")
    tracker._on_event(ctx2)
    assert "read_file" in tracker.tools_called
    assert "write_file" in tracker.tools_called
    assert TOOL_CALL_AFTER in tracker.hooks_emitted
    report = tracker.get_report()
    assert "tools_called" in report
    assert "read_file" in report["tools_called"]
    tracker.reset()
    assert len(tracker.tools_called) == 0


@pytest.mark.unit
def test_coverage_tracker_error_paths():
    """Error paths recorded (tool_error, llm_error)."""
    tracker = AgentCoverageTracker()
    tracker._on_event(HookContext(event=TOOL_CALL_ERROR, data={}, agent_id="a", run_id="r"))
    assert "tool_error" in tracker.error_paths


@pytest.mark.unit
def test_merge_coverage_reports():
    """Merge multiple coverage reports."""
    r1 = {"tools_called": ["a", "b"], "hooks_emitted": ["h1"], "error_paths": []}
    r2 = {"tools_called": ["b", "c"], "hooks_emitted": ["h2"], "error_paths": ["e1"]}
    merged = merge_coverage_reports([r1, r2])
    assert set(merged["tools_called"]) == {"a", "b", "c"}
    assert set(merged["hooks_emitted"]) == {"h1", "h2"}
    assert merged["error_paths"] == ["e1"]


@pytest.mark.unit
def test_coverage_tracker_print_report():
    """print_report runs without error."""
    tracker = AgentCoverageTracker()
    tracker._on_event(HookContext(event=TOOL_CALL_AFTER, data={"tool_name": "x"}, agent_id="a", run_id="r"))
    tracker.print_report(title="Test coverage")


@pytest.mark.unit
def test_merge_coverage_reports_empty():
    """merge_coverage_reports handles empty list."""
    merged = merge_coverage_reports([])
    assert merged["tools_called"] == []
    assert merged["hooks_emitted"] == []
    assert merged["error_paths"] == []
