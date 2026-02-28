"""
In-memory persistence implementation.

Useful for testing and development. All data is lost when the process ends.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.models.agent import AgentRun, AgentRunEvent, AgentLLMUsage, AgentRunStatus

logger = logging.getLogger(__name__)


class InMemoryPersistence(BasePersistence):
    """
    In-memory persistence implementation.

    Stores all data in memory. Useful for testing and development
    where persistence is not required.

    Example:
        >>> persistence = InMemoryPersistence()
        >>> agent = MyAgent("test-agent", persistence=persistence)
        >>> result = agent.run("Test objective")
        >>>
        >>> # Get run history
        >>> runs = persistence.get_agent_runs("test-agent")
        >>> print(f"Total runs: {len(runs)}")
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._runs: Dict[str, AgentRun] = {}
        self._events: Dict[str, List[AgentRunEvent]] = {}  # run_id -> events
        self._llm_usage: List[AgentLLMUsage] = []
        self._id_counter = 0

    def _next_id(self) -> int:
        """Generate next sequential ID."""
        self._id_counter += 1
        return self._id_counter

    # ==================== Agent Runs ====================

    def create_agent_run(self, run: AgentRun) -> None:
        """Create a new agent run record."""
        run.id = self._next_id()
        run.created_at = datetime.now()
        run.updated_at = datetime.now()
        self._runs[run.run_id] = run
        self._events[run.run_id] = []
        logger.debug(f"Created agent run: {run.run_id}")

    def update_agent_run(self, run_id: str, run: AgentRun) -> None:
        """Update an existing agent run record."""
        if run_id in self._runs:
            run.updated_at = datetime.now()
            self._runs[run_id] = run
            logger.debug(f"Updated agent run: {run_id}")
        else:
            logger.warning(f"Agent run not found for update: {run_id}")

    def get_agent_run(self, run_id: str) -> Optional[AgentRun]:
        """Get an agent run by ID."""
        return self._runs.get(run_id)

    def get_agent_runs(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AgentRun]:
        """Get agent runs with optional filtering."""
        runs = list(self._runs.values())

        # Filter by agent_id if provided
        if agent_id:
            runs = [r for r in runs if r.agent_id == agent_id]

        # Sort by created_at descending
        runs.sort(key=lambda r: r.created_at or datetime.min, reverse=True)

        # Apply pagination
        return runs[offset:offset + limit]

    def delete_agent_run(self, run_id: str) -> bool:
        """Delete an agent run."""
        if run_id in self._runs:
            del self._runs[run_id]
            if run_id in self._events:
                del self._events[run_id]
            logger.debug(f"Deleted agent run: {run_id}")
            return True
        return False

    # ==================== Agent Run Events ====================

    def log_agent_run_event(self, event: AgentRunEvent) -> None:
        """Log an agent run event."""
        event.id = self._next_id()
        event.created_at = datetime.now()

        if event.run_id not in self._events:
            self._events[event.run_id] = []

        self._events[event.run_id].append(event)
        logger.debug(f"Logged event {event.event_type} for run {event.run_id}")

    def get_agent_run_events(
        self,
        run_id: str,
        event_type: Optional[str] = None,
    ) -> List[AgentRunEvent]:
        """Get events for an agent run."""
        events = self._events.get(run_id, [])

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    # ==================== LLM Usage ====================

    def log_llm_usage(self, usage: AgentLLMUsage) -> None:
        """Log LLM usage for tracking."""
        usage.id = self._next_id()
        usage.created_at = datetime.now()
        self._llm_usage.append(usage)
        logger.debug(f"Logged LLM usage: {usage.provider}/{usage.model}")

    def get_llm_usage(
        self,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentLLMUsage]:
        """Get LLM usage records."""
        records = self._llm_usage

        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]

        if run_id:
            records = [r for r in records if r.run_id == run_id]

        # Sort by created_at descending
        records.sort(key=lambda r: r.created_at or datetime.min, reverse=True)

        return records[:limit]

    # ==================== Statistics ====================

    @staticmethod
    def _compute_percentiles(values: list[float], percentiles: list[int] | None = None) -> Dict[str, float]:
        """Compute percentile values from a list of numbers."""
        if not values:
            return {}
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]
        values_sorted = sorted(values)
        n = len(values_sorted)
        result = {}
        for p in percentiles:
            idx = (p / 100) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            result[f"p{p}"] = round(values_sorted[lower] * (1 - weight) + values_sorted[upper] * weight, 2)
        return result

    def get_agent_run_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for agent runs, including extended analytics."""
        runs = list(self._runs.values())

        if agent_id:
            runs = [r for r in runs if r.agent_id == agent_id]

        total_runs = len(runs)
        completed_runs = sum(1 for r in runs if r.status == AgentRunStatus.COMPLETED.value)
        error_runs = sum(1 for r in runs if r.status == AgentRunStatus.ERROR.value)

        total_iterations = sum(r.total_iterations for r in runs)
        avg_iterations = total_iterations / total_runs if total_runs > 0 else 0

        llm_usage = self._llm_usage
        if agent_id:
            llm_usage = [u for u in llm_usage if u.agent_id == agent_id]

        total_llm_calls = len(llm_usage)
        total_input_tokens = sum(u.input_tokens or 0 for u in llm_usage)
        total_output_tokens = sum(u.output_tokens or 0 for u in llm_usage)

        # Extended: tool metrics from events
        tool_metrics: Dict[str, Any] = {}
        all_events: List[AgentRunEvent] = []
        for events_list in self._events.values():
            all_events.extend(events_list)
        if agent_id:
            all_events = [e for e in all_events if e.agent_id == agent_id]

        for evt in all_events:
            if evt.event_type != "tool_call" or not evt.data:
                continue
            try:
                d = json.loads(evt.data)
            except (json.JSONDecodeError, TypeError):
                continue
            tname = d.get("tool_name", "unknown")
            if tname not in tool_metrics:
                tool_metrics[tname] = {"call_count": 0, "total_latency_ms": 0, "error_count": 0}
            tool_metrics[tname]["call_count"] += 1
            tool_metrics[tname]["total_latency_ms"] += d.get("latency_ms", 0)
            if d.get("error"):
                tool_metrics[tname]["error_count"] += 1
        for tname in tool_metrics:
            calls = tool_metrics[tname]["call_count"]
            tool_metrics[tname]["avg_latency_ms"] = (
                round(tool_metrics[tname]["total_latency_ms"] / calls, 2) if calls else 0
            )

        # Extended: latency percentiles
        latency_values = [float(u.latency_ms) for u in llm_usage if u.latency_ms]
        latency_percentiles = self._compute_percentiles(latency_values)

        # Token efficiency
        token_efficiency = (
            round(total_output_tokens / total_input_tokens, 4)
            if total_input_tokens > 0 else 0
        )
        avg_input_per_call = (
            round(total_input_tokens / total_llm_calls, 2)
            if total_llm_calls > 0 else 0
        )
        avg_output_per_call = (
            round(total_output_tokens / total_llm_calls, 2)
            if total_llm_calls > 0 else 0
        )

        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "error_runs": error_runs,
            "avg_iterations": round(avg_iterations, 2),
            "total_llm_calls": total_llm_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            # Extended analytics
            "tool_metrics": tool_metrics,
            "latency_percentiles": latency_percentiles,
            "token_efficiency": token_efficiency,
            "avg_input_tokens_per_call": avg_input_per_call,
            "avg_output_tokens_per_call": avg_output_per_call,
        }

    # ==================== Optional Methods ====================

    def clear_all(self) -> None:
        """Clear all stored data."""
        self._runs.clear()
        self._events.clear()
        self._llm_usage.clear()
        self._id_counter = 0
        logger.debug("Cleared all in-memory data")

    def get_all_data(self) -> Dict[str, Any]:
        """Get all stored data (for debugging/export)."""
        return {
            "runs": {k: v.to_dict() for k, v in self._runs.items()},
            "events": {k: [e.to_dict() for e in v] for k, v in self._events.items()},
            "llm_usage": [u.to_dict() for u in self._llm_usage],
        }

    # ==================== Audit Logs ====================

    def log_audit_event(self, event: Any) -> None:
        """Log an in-memory audit event (non-tamper-resistant; testing only)."""
        # Store on a synthetic attribute to avoid changing constructor signature.
        if not hasattr(self, "_audit_events"):
            self._audit_events: List[Dict[str, Any]] = []
        self._audit_events.append(dict(event))

    def get_audit_events(
        self,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return stored in-memory audit events."""
        events: List[Dict[str, Any]] = list(getattr(self, "_audit_events", []))
        if run_id is not None:
            events = [e for e in events if e.get("run_id") == run_id]
        if agent_id is not None:
            events = [e for e in events if e.get("agent_id") == agent_id]
        return events[:limit]
