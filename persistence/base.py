"""
Abstract base class for persistence implementations.

All persistence backends must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from curio_agent_sdk.core.models import AgentRun, AgentRunEvent, AgentLLMUsage


class BasePersistence(ABC):
    """
    Abstract base class for persistence implementations.

    All database backends (PostgreSQL, SQLite, in-memory) must implement
    this interface to be used with the Curio Agent SDK.

    Example:
        class MyPersistence(BasePersistence):
            def create_agent_run(self, run: AgentRun) -> None:
                # Your implementation
                pass
            # ... implement other methods

        # Use with agent
        persistence = MyPersistence()
        agent = MyAgent("agent-1", persistence=persistence)
    """

    # ==================== Agent Runs ====================

    @abstractmethod
    def create_agent_run(self, run: AgentRun) -> None:
        """
        Create a new agent run record.

        Args:
            run: AgentRun to create
        """
        pass

    @abstractmethod
    def update_agent_run(self, run_id: str, run: AgentRun) -> None:
        """
        Update an existing agent run record.

        Args:
            run_id: The run ID to update
            run: Updated AgentRun data
        """
        pass

    @abstractmethod
    def get_agent_run(self, run_id: str) -> Optional[AgentRun]:
        """
        Get an agent run by ID.

        Args:
            run_id: The run ID to retrieve

        Returns:
            AgentRun or None if not found
        """
        pass

    @abstractmethod
    def get_agent_runs(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AgentRun]:
        """
        Get agent runs with optional filtering.

        Args:
            agent_id: Filter by agent ID (optional)
            limit: Maximum number of runs to return
            offset: Offset for pagination

        Returns:
            List of AgentRun objects
        """
        pass

    @abstractmethod
    def delete_agent_run(self, run_id: str) -> bool:
        """
        Delete an agent run.

        Args:
            run_id: The run ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Agent Run Events ====================

    @abstractmethod
    def log_agent_run_event(self, event: AgentRunEvent) -> None:
        """
        Log an agent run event.

        Args:
            event: AgentRunEvent to log
        """
        pass

    @abstractmethod
    def get_agent_run_events(
        self,
        run_id: str,
        event_type: Optional[str] = None,
    ) -> List[AgentRunEvent]:
        """
        Get events for an agent run.

        Args:
            run_id: The run ID to get events for
            event_type: Filter by event type (optional)

        Returns:
            List of AgentRunEvent objects
        """
        pass

    # ==================== LLM Usage ====================

    @abstractmethod
    def log_llm_usage(self, usage: AgentLLMUsage) -> None:
        """
        Log LLM usage for tracking.

        Args:
            usage: AgentLLMUsage to log
        """
        pass

    @abstractmethod
    def get_llm_usage(
        self,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentLLMUsage]:
        """
        Get LLM usage records.

        Args:
            agent_id: Filter by agent ID (optional)
            run_id: Filter by run ID (optional)
            limit: Maximum number of records to return

        Returns:
            List of AgentLLMUsage objects
        """
        pass

    # ==================== Statistics ====================

    @abstractmethod
    def get_agent_run_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for agent runs.

        Args:
            agent_id: Filter by agent ID (optional)

        Returns:
            Dictionary with statistics:
            - total_runs: Total number of runs
            - completed_runs: Number of completed runs
            - error_runs: Number of failed runs
            - avg_iterations: Average iterations per run
            - total_llm_calls: Total LLM calls
        """
        pass

    # ==================== Optional Methods ====================

    def initialize_schema(self) -> None:
        """
        Initialize the database schema.

        This method should create all necessary tables if they don't exist.
        Default implementation does nothing.
        """
        pass

    def close(self) -> None:
        """
        Close any database connections.

        Default implementation does nothing.
        """
        pass

    def health_check(self) -> bool:
        """
        Check if the persistence layer is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return True
