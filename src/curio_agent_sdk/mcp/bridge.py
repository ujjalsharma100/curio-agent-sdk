"""
MCP bridge: connects to MCP servers at agent startup and registers their tools.

Implements Component so Runtime calls startup() before the first run and
shutdown() when the agent is closed. At startup, connects to each server,
discovers tools, and registers them with the agent's ToolRegistry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from curio_agent_sdk.core.component import Component
from curio_agent_sdk.core.circuit_breaker import CircuitBreaker
from curio_agent_sdk.mcp.client import MCPClient
from curio_agent_sdk.mcp.adapter import MCPToolAdapter
from curio_agent_sdk.mcp.config import MCPServerConfig

if TYPE_CHECKING:
    from curio_agent_sdk.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _describe_spec(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, MCPServerConfig):
        return spec.name or spec.url or f"{spec.command} ..."
    if isinstance(spec, dict):
        return spec.get("name") or spec.get("url") or spec.get("command") or "config"
    return "server"


class MCPBridge(Component):
    """
    Connects to MCP servers on startup and registers their tools with a ToolRegistry.

    Use with Agent/Builder via .mcp_server(url) or .mcp_server_config(config) or
    .mcp_servers_from_file(path). Each server can be a URL string or a
    Cursor/Claude-style config (command, args, env or url, headers).
    """

    def __init__(
        self,
        server_specs: list[str | dict[str, Any] | MCPServerConfig],
        tool_registry: "ToolRegistry",
        timeout: float = 30.0,
        resource_uris: list[str] | None = None,
        resolve_env: bool = True,
        circuit_breaker_max_failures: int = 3,
        circuit_breaker_recovery_seconds: float = 300.0,
    ):
        self.server_specs = list(server_specs)
        self.tool_registry = tool_registry
        self.timeout = timeout
        self.resource_uris = list(resource_uris or [])
        self.resolve_env = resolve_env
        self._clients: list[MCPClient] = []
        self._tool_names_added: list[str] = []
        # Per-server circuit breakers keyed by server description
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cb_max_failures = circuit_breaker_max_failures
        self._cb_recovery_seconds = circuit_breaker_recovery_seconds

    def get_circuit_breaker(self, server_desc: str) -> CircuitBreaker | None:
        """Get the circuit breaker for an MCP server by description."""
        return self._circuit_breakers.get(server_desc)

    async def startup(self) -> None:
        """Connect to each MCP server and register discovered tools."""
        for spec in self.server_specs:
            desc = _describe_spec(spec)
            cb = CircuitBreaker(
                max_failures=self._cb_max_failures,
                recovery_seconds=self._cb_recovery_seconds,
            )
            self._circuit_breakers[desc] = cb
            try:
                if isinstance(spec, str):
                    client = MCPClient(server_url=spec, timeout=self.timeout)
                else:
                    client = MCPClient(config=spec, timeout=self.timeout, resolve_env=self.resolve_env)
                await client.connect()
                cb.record_success()
                self._clients.append(client)
                tools = await client.list_all_tools()
                for t in tools:
                    name = t.name
                    if self.tool_registry.has(name):
                        name = f"mcp_{name}"
                    curio_tool = MCPToolAdapter.adapt(t, client, name_override=name)
                    self.tool_registry.register(curio_tool)
                    self._tool_names_added.append(name)
                logger.info("MCP: connected to %s, registered %d tools", desc, len(tools))
            except Exception as e:
                cb.record_failure()
                logger.warning("MCP: failed to connect to %s: %s", desc, e)
                raise

    async def shutdown(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("MCP disconnect error: %s", e)
        self._clients.clear()
        # Optionally remove tools we added (could break in-flight runs; leave registered)
        self._tool_names_added.clear()

    async def health_check(self) -> bool:
        """True if no MCP server circuit breakers are open."""
        for cb in self._circuit_breakers.values():
            if cb.is_open:
                return False
        return True

    async def get_resource_context(self, uris: list[str] | None = None) -> str:
        """
        Read the given resource URIs (or self.resource_uris) from connected
        MCP clients and return concatenated text for injection as context.

        Use with Runtime/agent to inject MCP resources into the conversation.
        """
        to_read = uris if uris is not None else self.resource_uris
        if not to_read or not self._clients:
            return ""
        parts: list[str] = []
        for uri in to_read:
            for client in self._clients:
                try:
                    content = await client.read_resource(uri)
                    if isinstance(content, str) and content.strip():
                        parts.append(content.strip())
                    break
                except Exception:
                    continue
        return "\n\n---\n\n".join(parts) if parts else ""
