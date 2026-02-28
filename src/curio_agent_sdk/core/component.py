"""
Component — base for SDK components that need lifecycle management.

Components can have startup (e.g. load indices, open connections),
shutdown (save state, close connections), and health checks.

Runtime calls startup before first use and shutdown when the agent is closed.
"""

from __future__ import annotations

from abc import ABC


class Component(ABC):
    """
    Base for all SDK components that need lifecycle management.

    Use this for:
    - Memory backends that load/save state (e.g. VectorMemory with persistence)
    - State stores that need to open/close connections
    - Persistence backends (schema init, connection close)
    - Any component that holds resources across runs

    Runtime calls startup() before the first run and shutdown() when the
    agent is closed (e.g. via agent.close() or async with agent).

    Example:
        class VectorMemory(Memory, Component):
            async def startup(self) -> None:
                await self._load_index()

            async def shutdown(self) -> None:
                await self._save_index()

            async def health_check(self) -> bool:
                return self._index is not None
    """

    async def startup(self) -> None:
        """
        Initialize the component (load state, open connections, etc.).

        Called once before the first use. Idempotent implementations are
        recommended so multiple calls are safe.
        """
        pass

    async def shutdown(self) -> None:
        """
        Clean up the component (save state, close connections, etc.).

        Called when the agent is closed. Implementations should be
        safe to call multiple times.
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the component is healthy and ready to use.

        Returns:
            True if the component is operational, False otherwise.
        """
        return True
