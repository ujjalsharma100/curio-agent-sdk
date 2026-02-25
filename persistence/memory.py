"""
In-memory persistence implementation.

Useful for testing and development. All data is lost when the process ends.
"""

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

    def get_agent_run_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for agent runs."""
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

        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "error_runs": error_runs,
            "avg_iterations": round(avg_iterations, 2),
            "total_llm_calls": total_llm_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
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
