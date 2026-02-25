"""
Abstract base class for persistence implementations.

Supports both sync and async patterns. Sync implementations work out of
the box. Async implementations can override the a* methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import asyncio

from curio_agent_sdk.models.agent import AgentRun, AgentRunEvent, AgentLLMUsage


class BasePersistence(ABC):
    """
    Abstract base class for persistence backends.

    Implementations:
    - SQLitePersistence: File-based (development)
    - PostgresPersistence: Production multi-user
    - InMemoryPersistence: Testing

    Sync methods are required. Async methods default to running
    sync methods in a thread.
    """

    # === Agent Runs ===

    @abstractmethod
    def create_agent_run(self, run: AgentRun) -> None:
        pass

    @abstractmethod
    def update_agent_run(self, run_id: str, run: AgentRun) -> None:
        pass

    @abstractmethod
    def get_agent_run(self, run_id: str) -> AgentRun | None:
        pass

    @abstractmethod
    def get_agent_runs(
        self,
        agent_id: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[AgentRun]:
        pass

    @abstractmethod
    def delete_agent_run(self, run_id: str) -> bool:
        pass

    # === Events ===

    @abstractmethod
    def log_agent_run_event(self, event: AgentRunEvent) -> None:
        pass

    @abstractmethod
    def get_agent_run_events(
        self,
        run_id: str,
        event_type: str | None = None,
    ) -> list[AgentRunEvent]:
        pass

    # === LLM Usage ===

    @abstractmethod
    def log_llm_usage(self, usage: AgentLLMUsage) -> None:
        pass

    @abstractmethod
    def get_llm_usage(
        self,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentLLMUsage]:
        pass

    # === Stats ===

    @abstractmethod
    def get_agent_run_stats(self, agent_id: str | None = None) -> dict[str, Any]:
        pass

    # === Lifecycle ===

    def initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        pass

    def close(self) -> None:
        """Close connections."""
        pass

    def health_check(self) -> bool:
        """Check if the backend is healthy."""
        return True

    # === Async wrappers (run sync in thread by default) ===

    async def acreate_agent_run(self, run: AgentRun) -> None:
        await asyncio.to_thread(self.create_agent_run, run)

    async def aupdate_agent_run(self, run_id: str, run: AgentRun) -> None:
        await asyncio.to_thread(self.update_agent_run, run_id, run)

    async def aget_agent_run(self, run_id: str) -> AgentRun | None:
        return await asyncio.to_thread(self.get_agent_run, run_id)

    async def alog_agent_run_event(self, event: AgentRunEvent) -> None:
        await asyncio.to_thread(self.log_agent_run_event, event)

    async def alog_llm_usage(self, usage: AgentLLMUsage) -> None:
        await asyncio.to_thread(self.log_llm_usage, usage)
