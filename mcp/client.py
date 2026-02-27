"""
MCP client: connect to MCP servers and call tools, list resources, list prompts.

Uses JSON-RPC 2.0 over stdio or HTTP. Implements initialize handshake and
tools/list, tools/call; resources/list, resources/read; prompts/list, prompts/get.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from curio_agent_sdk.mcp.transport import (
    MCPTransport,
    MCPError,
    transport_for_url,
    transport_from_config,
)
from curio_agent_sdk.mcp.config import MCPServerConfig, resolve_env_in_config

logger = logging.getLogger(__name__)

# MCP protocol version we support
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPTool:
    """MCP tool definition (from tools/list)."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    title: str | None = None


@dataclass
class MCPResource:
    """MCP resource (from resources/list)."""
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str | None = None


@dataclass
class MCPPrompt:
    """MCP prompt template (from prompts/list)."""
    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = field(default_factory=list)


class MCPClient:
    """
    Connect to an MCP server and expose its tools, resources, and prompts.

    Use connect() before calling list_tools/call_tool etc.; use disconnect() when done.

    Can be constructed from:
    - server_url (str): e.g. "stdio://npx -y @modelcontextprotocol/server-filesystem /tmp"
      or "http://localhost:8080"
    - config (dict | MCPServerConfig): Cursor/Claude-style config with command/args/env
      (stdio) or url/headers (HTTP). Set resolve_env=True to expand $VAR in env/headers.
    """

    def __init__(
        self,
        server_url: str | None = None,
        *,
        config: dict[str, Any] | MCPServerConfig | None = None,
        timeout: float = 30.0,
        resolve_env: bool = True,
    ):
        if config is not None:
            cfg = config if isinstance(config, MCPServerConfig) else MCPServerConfig.from_dict(config)
            if resolve_env:
                cfg = resolve_env_in_config(cfg)
            self._transport = transport_from_config(cfg)
            self.server_url = cfg.url or f"stdio://{cfg.command} {' '.join(cfg.args)}"
        elif server_url:
            self.server_url = server_url
            self._transport = transport_for_url(server_url, timeout=timeout)
        else:
            raise ValueError("Provide either server_url or config")
        self._initialized = False

    async def connect(self) -> None:
        """Connect and perform MCP initialize handshake."""
        await self._transport.connect()
        result = await self._transport.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "curio-agent-sdk", "version": "0.6.0"},
            },
        )
        self._initialized = True
        logger.debug("MCP initialized: %s", result.get("serverInfo", result))
        # Some servers expect a subsequent "initialized" notification (no response)
        try:
            await self._send_notification("notifications/initialized", {})
        except Exception as e:
            logger.debug("MCP notifications/initialized failed (optional): %s", e)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification (no id, no response expected). Stdio transport only."""
        import json
        writer = getattr(self._transport, "_writer", None)
        if writer is None or (getattr(writer, "is_closing", lambda: False)()):
            return
        msg = (json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n").encode("utf-8")
        writer.write(msg)
        await writer.drain()

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._initialized = False
        await self._transport.disconnect()

    async def list_tools(self, cursor: str | None = None) -> tuple[list[MCPTool], str | None]:
        """
        List tools offered by the server.

        Returns:
            (list of tools, next_cursor or None)
        """
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        result = await self._transport.request("tools/list", params if params else None)
        tools_data = result.get("tools", [])
        next_cursor = result.get("nextCursor")
        tools = [
            MCPTool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                title=t.get("title"),
            )
            for t in tools_data
        ]
        return tools, next_cursor

    async def list_all_tools(self) -> list[MCPTool]:
        """List all tools (handles pagination)."""
        all_tools: list[MCPTool] = []
        cursor: str | None = None
        while True:
            batch, cursor = await self.list_tools(cursor=cursor)
            all_tools.extend(batch)
            if not cursor:
                break
        return all_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Invoke a tool by name with the given arguments.

        Returns the tool result. Content is typically a list of content items
        (text, image, etc.); we return a simplified string or structured result
        for agent consumption.
        """
        result = await self._transport.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        content = result.get("content", [])
        is_error = result.get("isError", False)
        # Flatten text content for the agent
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif "text" in item:
                    texts.append(str(item["text"]))
        out = "\n".join(texts).strip() if texts else str(result)
        if is_error:
            from curio_agent_sdk.models.exceptions import ToolExecutionError
            raise ToolExecutionError(name, RuntimeError(out))
        # Also return structuredContent if present (e.g. for JSON output)
        if "structuredContent" in result:
            return result.get("structuredContent")
        return out

    async def list_resources(self, cursor: str | None = None) -> tuple[list[MCPResource], str | None]:
        """List resources. Returns (resources, next_cursor)."""
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        try:
            result = await self._transport.request("resources/list", params if params else None)
        except MCPError as e:
            if "not supported" in str(e).lower() or "unknown method" in str(e).lower():
                return [], None
            raise
        resources_data = result.get("resources", [])
        next_cursor = result.get("nextCursor")
        resources = [
            MCPResource(
                uri=r.get("uri", ""),
                name=r.get("name", ""),
                description=r.get("description", ""),
                mime_type=r.get("mimeType"),
            )
            for r in resources_data
        ]
        return resources, next_cursor

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI. Returns content (e.g. text, base64 blob)."""
        result = await self._transport.request("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        texts = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text", ""))
        return "\n".join(texts) if texts else result

    async def list_prompts(self, cursor: str | None = None) -> tuple[list[MCPPrompt], str | None]:
        """List prompts. Returns (prompts, next_cursor)."""
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        try:
            result = await self._transport.request("prompts/list", params if params else None)
        except MCPError as e:
            if "not supported" in str(e).lower() or "unknown method" in str(e).lower():
                return [], None
            raise
        prompts_data = result.get("prompts", [])
        next_cursor = result.get("nextCursor")
        prompts = [
            MCPPrompt(
                name=p.get("name", ""),
                description=p.get("description", ""),
                arguments=p.get("arguments", []),
            )
            for p in prompts_data
        ]
        return prompts, next_cursor

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Get a prompt by name with optional arguments. Returns messages or content."""
        params: dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        result = await self._transport.request("prompts/get", params)
        return result
