"""
Adapt MCP tools and resources to Curio SDK types.

MCPToolAdapter converts MCP tool definitions to Curio Tool instances
that call back into the MCP client.
"""

from __future__ import annotations

from typing import Any

from curio_agent_sdk.core.tools.tool import Tool, ToolConfig
from curio_agent_sdk.core.tools.schema import ToolSchema

from curio_agent_sdk.mcp.client import MCPClient, MCPTool


class MCPToolAdapter:
    """Adapts MCP tools to Curio SDK Tool format."""

    @staticmethod
    def adapt(mcp_tool: MCPTool, client: MCPClient, name_override: str | None = None) -> Tool:
        """
        Convert an MCP tool definition to a Curio Tool.

        The returned Tool's execute() calls client.call_tool(name, arguments).
        Use name_override to avoid registry clashes (e.g. "mcp_read_file").
        """
        name = name_override or mcp_tool.name
        description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        input_schema = mcp_tool.input_schema or {"type": "object", "properties": {}}

        schema = ToolSchema.from_json_schema(name, description, input_schema)

        async def _execute(**kwargs: Any) -> Any:
            return await client.call_tool(mcp_tool.name, kwargs)

        return Tool(
            func=_execute,
            name=name,
            description=description,
            schema=schema,
            config=ToolConfig(timeout=60.0),
        )

    @staticmethod
    def adapt_all(
        mcp_tools: list[MCPTool],
        client: MCPClient,
        name_prefix: str | None = None,
    ) -> list[Tool]:
        """Convert a list of MCP tools to Curio Tools. Optional name_prefix for all."""
        return [
            MCPToolAdapter.adapt(
                t,
                client,
                name_override=f"{name_prefix}{t.name}" if name_prefix else None,
            )
            for t in mcp_tools
        ]
