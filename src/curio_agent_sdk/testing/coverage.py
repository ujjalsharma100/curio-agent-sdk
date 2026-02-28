"""
Test coverage reporting for agent-specific metrics.

Tracks tool coverage (which tools were called), hook coverage (which lifecycle
events were emitted), and error-path coverage (run_error, tool_error, llm_error,
timeout, cancelled) during test runs. Register with an agent's HookRegistry
before running tests, then call get_report() to inspect or assert on coverage.

Example:
    from curio_agent_sdk.testing.coverage import AgentCoverageTracker
    from curio_agent_sdk.core.hooks import HookRegistry

    registry = HookRegistry()
    coverage = AgentCoverageTracker()
    coverage.register(registry)

    agent = Agent.builder().hook_registry(registry).tools([...]).build()
    await agent.arun("Do something")

    report = coverage.get_report()
    assert "read_file" in report["tools_called"]
    assert "agent.run.after" in report["hooks_emitted"]
    coverage.print_report()
"""

from __future__ import annotations

from typing import Any

from curio_agent_sdk.core.hooks import (
    AGENT_RUN_ERROR,
    HOOK_EVENTS,
    LLM_CALL_ERROR,
    TOOL_CALL_AFTER,
    TOOL_CALL_ERROR,
    HookContext,
    HookRegistry,
)


class AgentCoverageTracker:
    """
    Tracks agent-specific coverage during test runs: tools called, hooks emitted,
    and error paths hit. Register with a HookRegistry to record metrics.
    """

    def __init__(self) -> None:
        self._tools_called: set[str] = set()
        self._hooks_emitted: set[str] = set()
        self._error_paths: set[str] = set()

    def register(self, registry: HookRegistry) -> None:
        """
        Register this tracker with the given HookRegistry so all hook events
        and relevant tool/error data are recorded.
        """
        for event in HOOK_EVENTS:
            registry.on(event, self._on_event, priority=1000)  # low priority, run last

    def _on_event(self, ctx: HookContext) -> None:
        self._hooks_emitted.add(ctx.event)
        data = ctx.data or {}
        if ctx.event == TOOL_CALL_AFTER:
            tool_name = data.get("tool_name") or data.get("tool")
            if tool_name:
                self._tools_called.add(str(tool_name))
        elif ctx.event == TOOL_CALL_ERROR:
            self._error_paths.add("tool_error")
        elif ctx.event == AGENT_RUN_ERROR:
            kind = data.get("error_kind") or "run_error"
            self._error_paths.add(str(kind))
        elif ctx.event == LLM_CALL_ERROR:
            self._error_paths.add("llm_error")

    def reset(self) -> None:
        """Clear all recorded coverage (e.g. between tests)."""
        self._tools_called.clear()
        self._hooks_emitted.clear()
        self._error_paths.clear()

    @property
    def tools_called(self) -> set[str]:
        """Set of tool names that were called."""
        return set(self._tools_called)

    @property
    def hooks_emitted(self) -> set[str]:
        """Set of hook event names that were emitted."""
        return set(self._hooks_emitted)

    @property
    def error_paths(self) -> set[str]:
        """Set of error path identifiers hit (e.g. run_error, tool_error, timeout)."""
        return set(self._error_paths)

    def get_report(self) -> dict[str, Any]:
        """Return a dict of coverage metrics for assertions or logging."""
        return {
            "tools_called": sorted(self._tools_called),
            "hooks_emitted": sorted(self._hooks_emitted),
            "error_paths": sorted(self._error_paths),
        }

    def print_report(self, title: str = "Agent coverage") -> None:
        """Print a human-readable coverage report to stdout."""
        lines = [f"{title}", "-" * 40]
        lines.append("Tools called: " + ", ".join(sorted(self._tools_called)) or "(none)")
        lines.append("Hooks emitted: " + ", ".join(sorted(self._hooks_emitted)) or "(none)")
        lines.append("Error paths: " + ", ".join(sorted(self._error_paths)) or "(none)")
        print("\n".join(lines))


def merge_coverage_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge multiple coverage reports (e.g. from several test runs) into one.

    Each report is expected to have keys tools_called, hooks_emitted, error_paths
    (list or set). Returns a single report with merged sorted lists.
    """
    tools: set[str] = set()
    hooks: set[str] = set()
    errors: set[str] = set()
    for r in reports:
        tools.update(r.get("tools_called", []))
        hooks.update(r.get("hooks_emitted", []))
        errors.update(r.get("error_paths", []))
    return {
        "tools_called": sorted(tools),
        "hooks_emitted": sorted(hooks),
        "error_paths": sorted(errors),
    }
