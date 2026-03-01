"""
Abstract base class for persistence implementations.

Supports both sync and async patterns. Sync implementations work out of
the box. Async implementations can override the a* methods.

Implements Component for lifecycle (startup = initialize_schema,
shutdown = close, health_check).
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any

from curio_agent_sdk.base import Component
from curio_agent_sdk.models.agent import AgentRun, AgentRunEvent, AgentLLMUsage


class BasePersistence(Component):
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

    # === Audit Logs (optional; tamper-evident chain) ===

    def log_audit_event(self, event: Any) -> None:
        """
        Persist a structured audit log event.

        Concrete implementations should store enough information to make the
        log tamper-evident (e.g. hash chain via prev_hash/hash fields).
        The event type is intentionally left generic here to avoid a hard
        dependency on a specific audit model in the core layer.
        """
        # Default: no-op for backends that don't support audit logging.
        return None

    def get_audit_events(
        self,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """
        Retrieve audit log events, optionally filtered by run_id / agent_id.

        Concrete implementations should return backend-specific audit event
        objects or dictionaries.
        """
        return []

    # === Lifecycle (sync; Component async methods wrap these) ===

    def initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        pass

    def close(self) -> None:
        """Close connections."""
        pass

    def health_check(self) -> bool:
        """Check if the backend is healthy (sync). Subclasses may override."""
        return True

    # === Component lifecycle (async wrappers for Runtime) ===

    async def startup(self) -> None:
        """Initialize schema (async wrapper)."""
        await asyncio.to_thread(self.initialize_schema)

    async def shutdown(self) -> None:
        """Close connections (async wrapper)."""
        await asyncio.to_thread(self.close)

    # health_check: use Component default (True). Sync health_check() above
    # remains for backward compat (e.g. observability backend).

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

    async def alog_audit_event(self, event: Any) -> None:
        """Async wrapper for log_audit_event."""
        await asyncio.to_thread(self.log_audit_event, event)
