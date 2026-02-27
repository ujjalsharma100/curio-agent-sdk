"""
ConnectorBridge — manages connector lifecycle and registers connector tools with the agent.

Implements Component so Runtime calls startup() before the first run and
shutdown() when the agent is closed. At startup, connects each connector,
registers its tools (with optional name prefix to avoid clashes), and
aggregates resources for context injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from curio_agent_sdk.core.component import Component
from curio_agent_sdk.core.circuit_breaker import CircuitBreaker
from curio_agent_sdk.connectors.base import Connector, ConnectorResource

if TYPE_CHECKING:
    from curio_agent_sdk.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ConnectorBridge(Component):
    """
    Manages a set of connectors: lifecycle (connect/disconnect) and tool registration.

    Each connector gets a CircuitBreaker that tracks health. When a connector's
    circuit is open, its tools return an error instead of calling the connector.

    Use with Agent/Builder via .connector(connector_instance). Each connector's
    tools are registered with the agent's ToolRegistry; on name clash, tools
    are prefixed with the connector name (e.g. github_create_pr).
    """

    def __init__(
        self,
        connectors: list[Connector],
        tool_registry: "ToolRegistry",
        resolve_credentials: bool = True,
        circuit_breaker_max_failures: int = 3,
        circuit_breaker_recovery_seconds: float = 300.0,
    ):
        self.connectors = list(connectors)
        self.tool_registry = tool_registry
        self.resolve_credentials = resolve_credentials
        self._tool_names_added: list[str] = []
        # Per-connector circuit breakers keyed by connector name
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cb_max_failures = circuit_breaker_max_failures
        self._cb_recovery_seconds = circuit_breaker_recovery_seconds

    def get_circuit_breaker(self, connector_name: str) -> CircuitBreaker | None:
        """Get the circuit breaker for a connector by name."""
        return self._circuit_breakers.get(connector_name)

    async def startup(self) -> None:
        """Connect each connector and register its tools."""
        from curio_agent_sdk.connectors.base import resolve_credentials as resolve_creds

        for conn in self.connectors:
            conn_name = conn.name or type(conn).__name__
            cb = CircuitBreaker(
                max_failures=self._cb_max_failures,
                recovery_seconds=self._cb_recovery_seconds,
            )
            self._circuit_breakers[conn_name] = cb
            try:
                creds = getattr(conn, "_credentials", None)
                if creds is not None and self.resolve_credentials and isinstance(creds, dict):
                    creds = resolve_creds(creds)
                await conn.connect(credentials=creds)
                cb.record_success()
                tools = conn.get_tools()
                prefix = f"{conn.name}_" if conn.name else ""
                for t in tools:
                    name = t.name
                    if self.tool_registry.has(name):
                        name = f"{prefix}{name}"
                    if name != t.name:
                        from curio_agent_sdk.core.tools.tool import Tool
                        t = Tool(
                            func=t.func,
                            name=name,
                            description=t.description,
                            schema=t.schema,
                            config=t.config,
                        )
                    self.tool_registry.register(t)
                    self._tool_names_added.append(t.name)
                logger.info(
                    "Connector %s: connected, registered %d tools",
                    conn_name,
                    len(tools),
                )
            except Exception as e:
                cb.record_failure()
                logger.warning("Connector %s startup failed: %s", conn_name, e)
                raise

    async def shutdown(self) -> None:
        """Disconnect all connectors."""
        for conn in self.connectors:
            try:
                await conn.disconnect()
            except Exception as e:
                logger.debug("Connector %s disconnect error: %s", conn.name or type(conn).__name__, e)
        self._tool_names_added.clear()

    async def health_check(self) -> bool:
        """True if all connectors report healthy and no circuits are open."""
        for conn in self.connectors:
            conn_name = conn.name or type(conn).__name__
            cb = self._circuit_breakers.get(conn_name)
            if cb and cb.is_open:
                return False
            try:
                if not await conn.health_check():
                    return False
            except Exception:
                return False
        return True

    async def get_resource_context(self) -> str:
        """
        Aggregate all connectors' get_resources() into a single context string.

        Used by Runtime to inject connector context into the system message.
        """
        parts: list[str] = []
        for conn in self.connectors:
            resources = conn.get_resources()
            if not resources:
                continue
            for r in resources:
                if isinstance(r, ConnectorResource):
                    parts.append(f"## {r.uri}\n\n{r.content}")
                else:
                    parts.append(str(r))
        if not parts:
            return ""
        return "\n\n---\n\n".join(parts)
