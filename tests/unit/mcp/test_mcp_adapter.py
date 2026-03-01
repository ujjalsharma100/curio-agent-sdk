"""
Unit tests for MCPToolAdapter — adapt single MCPTool to Tool, adapt_all, schema mapping.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from curio_agent_sdk.mcp.client import MCPClient, MCPTool
from curio_agent_sdk.mcp.adapter import MCPToolAdapter
from curio_agent_sdk.core.tools.tool import Tool


@pytest.fixture
def mock_mcp_client():
    client = MagicMock(spec=MCPClient)
    client.call_tool = AsyncMock(return_value="ok")
    return client


@pytest.mark.unit
def test_tool_from_mcp(mock_mcp_client):
    """Convert single MCPTool to Curio Tool; execute calls client.call_tool."""
    mcp_tool = MCPTool(
        name="read_file",
        description="Read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    tool = MCPToolAdapter.adapt(mcp_tool, mock_mcp_client)
    assert isinstance(tool, Tool)
    assert tool.name == "read_file"
    assert tool.description == "Read a file"
    assert "path" in [p.name for p in (tool.schema.parameters or [])]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_from_mcp_execute(mock_mcp_client):
    """Adapted tool execute calls client.call_tool with correct name and kwargs."""
    mcp_tool = MCPTool(
        name="echo",
        description="Echo",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    tool = MCPToolAdapter.adapt(mcp_tool, mock_mcp_client)
    result = await tool.execute(text="hello")
    mock_mcp_client.call_tool.assert_called_once_with("echo", {"text": "hello"})
    assert result == "ok"


@pytest.mark.unit
def test_tool_from_mcp_name_override(mock_mcp_client):
    """name_override is used for the Tool name."""
    mcp_tool = MCPTool(name="read_file", description="Read", input_schema={})
    tool = MCPToolAdapter.adapt(mcp_tool, mock_mcp_client, name_override="mcp_read_file")
    assert tool.name == "mcp_read_file"


@pytest.mark.unit
def test_tools_from_mcp(mock_mcp_client):
    """adapt_all converts list of MCPTools to Curio Tools."""
    mcp_tools = [
        MCPTool(name="a", description="A", input_schema={}),
        MCPTool(name="b", description="B", input_schema={}),
    ]
    tools = MCPToolAdapter.adapt_all(mcp_tools, mock_mcp_client)
    assert len(tools) == 2
    assert [t.name for t in tools] == ["a", "b"]


@pytest.mark.unit
def test_tools_from_mcp_with_prefix(mock_mcp_client):
    """adapt_all with name_prefix prefixes all names."""
    mcp_tools = [MCPTool(name="tool1", description="", input_schema={})]
    tools = MCPToolAdapter.adapt_all(mcp_tools, mock_mcp_client, name_prefix="mcp_")
    assert tools[0].name == "mcp_tool1"


@pytest.mark.unit
def test_adapter_schema_mapping(mock_mcp_client):
    """Schema properties and required are mapped to ToolSchema."""
    mcp_tool = MCPTool(
        name="search",
        description="Search",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    tool = MCPToolAdapter.adapt(mcp_tool, mock_mcp_client)
    assert tool.schema is not None
    assert tool.schema.name == "search"
    param_names = [p.name for p in (tool.schema.parameters or [])]
    assert "query" in param_names
    assert "limit" in param_names
