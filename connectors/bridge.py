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
from curio_agent_sdk.connectors.base import Connector, ConnectorResource

if TYPE_CHECKING:
    from curio_agent_sdk.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ConnectorBridge(Component):
    """
    Manages a set of connectors: lifecycle (connect/disconnect) and tool registration.

    Use with Agent/Builder via .connector(connector_instance). Each connector's
    tools are registered with the agent's ToolRegistry; on name clash, tools
    are prefixed with the connector name (e.g. github_create_pr).
    """

    def __init__(
        self,
        connectors: list[Connector],
        tool_registry: "ToolRegistry",
        resolve_credentials: bool = True,
    ):
        self.connectors = list(connectors)
        self.tool_registry = tool_registry
        self.resolve_credentials = resolve_credentials
        self._tool_names_added: list[str] = []

    async def startup(self) -> None:
        """Connect each connector and register its tools."""
        from curio_agent_sdk.connectors.base import resolve_credentials as resolve_creds

        for conn in self.connectors:
            try:
                creds = getattr(conn, "_credentials", None)
                if creds is not None and self.resolve_credentials and isinstance(creds, dict):
                    creds = resolve_creds(creds)
                await conn.connect(credentials=creds)
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
                    conn.name or type(conn).__name__,
                    len(tools),
                )
            except Exception as e:
                logger.warning("Connector %s startup failed: %s", conn.name or type(conn).__name__, e)
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
        """True if all connectors report healthy."""
        for conn in self.connectors:
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
