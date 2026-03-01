"""
Unit tests for MCPClient — creation, connect, disconnect, list_tools, call_tool,
list_resources, read_resource, list_prompts, get_prompt, timeout handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from curio_agent_sdk.mcp.client import (
    MCPClient,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCP_PROTOCOL_VERSION,
)
from curio_agent_sdk.mcp.transport import MCPError
from curio_agent_sdk.models.exceptions import ToolExecutionError


@pytest.fixture
def mock_transport():
    """Transport that can be configured for each test."""
    transport = MagicMock()
    transport.connect = AsyncMock()
    transport.disconnect = AsyncMock()
    transport.request = AsyncMock()
    return transport


@pytest.fixture
def client_with_mock_transport(mock_transport):
    """MCPClient with injected mock transport (bypass __init__ URL/config)."""
    client = object.__new__(MCPClient)
    client._transport = mock_transport
    client._initialized = False
    client.server_url = "http://test"
    return client


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_client_creation_with_url():
    """Constructor with server_url creates client with transport for URL."""
    client = MCPClient(server_url="http://localhost:8080")
    assert client.server_url == "http://localhost:8080"
    assert client._transport is not None
    assert client._initialized is False


@pytest.mark.unit
def test_mcp_client_creation_with_stdio_url():
    """Constructor with stdio URL is accepted."""
    client = MCPClient(server_url="stdio://npx -y @mcp/server")
    assert "stdio://" in client.server_url
    assert client._transport is not None


@pytest.mark.unit
def test_mcp_client_creation_requires_url_or_config():
    """Constructor raises if neither server_url nor config provided."""
    with pytest.raises(ValueError, match="Provide either server_url or config"):
        MCPClient()


@pytest.mark.unit
def test_mcp_client_creation_with_config():
    """Constructor with config creates client (stdio or http from config)."""
    config = {"url": "http://localhost:8080", "headers": {}}
    client = MCPClient(config=config)
    assert client.server_url == "http://localhost:8080"
    assert client._transport is not None


# ---------------------------------------------------------------------------
# Connect / Disconnect
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_connect(client_with_mock_transport, mock_transport):
    """Connect performs handshake and sets _initialized."""
    mock_transport.request.return_value = {"serverInfo": {"name": "test"}}
    await client_with_mock_transport.connect()
    mock_transport.connect.assert_called_once()
    mock_transport.request.assert_called()
    call_args = mock_transport.request.call_args_list[0]
    assert call_args[0][0] == "initialize"
    assert call_args[0][1]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert client_with_mock_transport._initialized is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_disconnect(client_with_mock_transport, mock_transport):
    """Disconnect clears state and calls transport disconnect."""
    client_with_mock_transport._initialized = True
    await client_with_mock_transport.disconnect()
    assert client_with_mock_transport._initialized is False
    mock_transport.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# list_tools / list_all_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_tools(client_with_mock_transport, mock_transport):
    """List tools returns MCPTool list and optional cursor."""
    mock_transport.request.return_value = {
        "tools": [
            {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}},
        ],
        "nextCursor": None,
    }
    tools, cursor = await client_with_mock_transport.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "read_file"
    assert tools[0].description == "Read a file"
    assert cursor is None
    mock_transport.request.assert_called_once_with("tools/list", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_tools_with_cursor(client_with_mock_transport, mock_transport):
    """List tools with cursor passes cursor to request."""
    mock_transport.request.return_value = {"tools": [], "nextCursor": "page2"}
    _, next_cursor = await client_with_mock_transport.list_tools(cursor="page1")
    mock_transport.request.assert_called_once_with("tools/list", {"cursor": "page1"})
    assert next_cursor == "page2"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_all_tools(client_with_mock_transport, mock_transport):
    """List all tools paginates until no cursor."""
    mock_transport.request.side_effect = [
        {"tools": [{"name": "a", "description": "", "inputSchema": {}}], "nextCursor": "c1"},
        {"tools": [{"name": "b", "description": "", "inputSchema": {}}], "nextCursor": None},
    ]
    all_tools = await client_with_mock_transport.list_all_tools()
    assert len(all_tools) == 2
    assert [t.name for t in all_tools] == ["a", "b"]


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_call_tool(client_with_mock_transport, mock_transport):
    """Call tool sends tools/call and returns flattened text content."""
    mock_transport.request.return_value = {
        "content": [{"type": "text", "text": "file contents"}],
        "isError": False,
    }
    result = await client_with_mock_transport.call_tool("read_file", {"path": "/tmp/x"})
    assert result == "file contents"
    mock_transport.request.assert_called_once_with(
        "tools/call",
        {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_call_tool_structured_content(client_with_mock_transport, mock_transport):
    """Call tool returns structuredContent when present."""
    mock_transport.request.return_value = {
        "content": [],
        "isError": False,
        "structuredContent": {"key": "value"},
    }
    result = await client_with_mock_transport.call_tool("tool", {})
    assert result == {"key": "value"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_call_tool_error_raises(client_with_mock_transport, mock_transport):
    """Call tool with isError raises ToolExecutionError."""
    mock_transport.request.return_value = {
        "content": [{"type": "text", "text": "Permission denied"}],
        "isError": True,
    }
    with pytest.raises(ToolExecutionError):
        await client_with_mock_transport.call_tool("read_file", {})


# ---------------------------------------------------------------------------
# list_resources / read_resource
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_resources(client_with_mock_transport, mock_transport):
    """List resources returns MCPResource list."""
    mock_transport.request.return_value = {
        "resources": [
            {"uri": "file:///a", "name": "A", "description": "Resource A"},
        ],
        "nextCursor": None,
    }
    resources, cursor = await client_with_mock_transport.list_resources()
    assert len(resources) == 1
    assert resources[0].uri == "file:///a"
    assert resources[0].name == "A"
    assert cursor is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_resources_not_supported(client_with_mock_transport, mock_transport):
    """List resources returns empty when server does not support it."""
    mock_transport.request.side_effect = MCPError("Method not supported")
    resources, cursor = await client_with_mock_transport.list_resources()
    assert resources == []
    assert cursor is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_read_resource(client_with_mock_transport, mock_transport):
    """Read resource returns concatenated text content."""
    mock_transport.request.return_value = {
        "contents": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
    }
    result = await client_with_mock_transport.read_resource("file:///x")
    assert result == "hello\nworld"
    mock_transport.request.assert_called_once_with("resources/read", {"uri": "file:///x"})


# ---------------------------------------------------------------------------
# list_prompts / get_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_prompts(client_with_mock_transport, mock_transport):
    """List prompts returns MCPPrompt list."""
    mock_transport.request.return_value = {
        "prompts": [{"name": "summarize", "description": "Summarize", "arguments": []}],
        "nextCursor": None,
    }
    prompts, cursor = await client_with_mock_transport.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].name == "summarize"
    assert cursor is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_list_prompts_not_supported(client_with_mock_transport, mock_transport):
    """List prompts returns empty when not supported."""
    mock_transport.request.side_effect = MCPError("Unknown method")
    prompts, cursor = await client_with_mock_transport.list_prompts()
    assert prompts == []
    assert cursor is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_client_get_prompt(client_with_mock_transport, mock_transport):
    """Get prompt returns result from server."""
    mock_transport.request.return_value = {"messages": [{"role": "user", "content": "Hi"}]}
    result = await client_with_mock_transport.get_prompt("greet", {"name": "Alice"})
    assert result == {"messages": [{"role": "user", "content": "Hi"}]}
    mock_transport.request.assert_called_once()
    call_args = mock_transport.request.call_args[0]
    assert call_args[0] == "prompts/get"
    assert call_args[1]["name"] == "greet"
    assert call_args[1]["arguments"] == {"name": "Alice"}


# ---------------------------------------------------------------------------
# MCPTool / MCPResource / MCPPrompt dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_tool_dataclass():
    """MCPTool has name, description, input_schema, title."""
    t = MCPTool(name="x", description="d", input_schema={"type": "object"}, title="X")
    assert t.name == "x"
    assert t.description == "d"
    assert t.input_schema == {"type": "object"}
    assert t.title == "X"


@pytest.mark.unit
def test_mcp_resource_dataclass():
    """MCPResource has uri, name, description, mime_type."""
    r = MCPResource(uri="file:///a", name="A", description="d", mime_type="text/plain")
    assert r.uri == "file:///a"
    assert r.name == "A"
    assert r.mime_type == "text/plain"


@pytest.mark.unit
def test_mcp_prompt_dataclass():
    """MCPPrompt has name, description, arguments."""
    p = MCPPrompt(name="p", description="d", arguments=[{"name": "x"}])
    assert p.name == "p"
    assert p.arguments == [{"name": "x"}]
