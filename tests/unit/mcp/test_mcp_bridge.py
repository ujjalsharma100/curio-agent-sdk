"""
Unit tests for MCPBridge — startup connects to servers and registers tools,
shutdown disconnects, get_tools returns adapted tools, health_check, get_resource_context.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from curio_agent_sdk.mcp.bridge import MCPBridge, _describe_spec
from curio_agent_sdk.mcp.client import MCPClient, MCPTool
from curio_agent_sdk.mcp.config import MCPServerConfig
from curio_agent_sdk.core.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def mock_mcp_client():
    client = MagicMock(spec=MCPClient)
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.list_all_tools = AsyncMock(return_value=[
        MCPTool(name="tool_a", description="A", input_schema={}),
    ])
    client.read_resource = AsyncMock(return_value="resource content")
    return client


@pytest.mark.unit
def test_describe_spec_string():
    """_describe_spec for URL string returns the string (or truncated)."""
    assert _describe_spec("stdio://npx server") == "stdio://npx server"


@pytest.mark.unit
def test_describe_spec_config():
    """_describe_spec for MCPServerConfig uses name or url or command."""
    cfg = MCPServerConfig(name="github", command="npx", args=["server"])
    assert _describe_spec(cfg) == "github"
    cfg2 = MCPServerConfig(name="", url="https://mcp.example.com")
    assert _describe_spec(cfg2) == "https://mcp.example.com"


@pytest.mark.unit
def test_describe_spec_dict():
    """_describe_spec for dict uses name, url, or command."""
    assert _describe_spec({"name": "x"}) == "x"
    assert _describe_spec({"url": "http://x"}) == "http://x"
    assert _describe_spec({"command": "npx"}) == "npx"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_startup(tool_registry, mock_mcp_client):
    """Startup connects to MCP server and registers tools (with patched client creation)."""
    with patch(
        "curio_agent_sdk.mcp.bridge.MCPClient",
        return_value=mock_mcp_client,
    ):
        bridge = MCPBridge(
            server_specs=["http://localhost:9999"],
            tool_registry=tool_registry,
        )
        await bridge.startup()
    mock_mcp_client.connect.assert_called_once()
    mock_mcp_client.list_all_tools.assert_called_once()
    assert tool_registry.has("tool_a") or tool_registry.has("mcp_tool_a")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_shutdown(tool_registry, mock_mcp_client):
    """Shutdown disconnects all clients."""
    bridge = MCPBridge(server_specs=[], tool_registry=tool_registry)
    bridge._clients = [mock_mcp_client]
    await bridge.shutdown()
    mock_mcp_client.disconnect.assert_called_once()
    assert len(bridge._clients) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_get_tools_via_registry(tool_registry, mock_mcp_client):
    """After startup, tools are in registry (get_tools = tools registered)."""
    with patch(
        "curio_agent_sdk.mcp.bridge.MCPClient",
        return_value=mock_mcp_client,
    ):
        bridge = MCPBridge(
            server_specs=["http://test"],
            tool_registry=tool_registry,
        )
        await bridge.startup()
    # Bridge registers tools on registry; we don't have get_tools on bridge, we check registry
    all_names = list(tool_registry._tools.keys())
    assert len(all_names) >= 1
    assert "tool_a" in all_names or any("tool_a" in n for n in all_names)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_health_check(tool_registry):
    """health_check returns True when no circuit breakers are open."""
    bridge = MCPBridge(server_specs=[], tool_registry=tool_registry)
    assert await bridge.health_check() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_get_resource_context(tool_registry, mock_mcp_client):
    """get_resource_context reads URIs from clients and concatenates."""
    bridge = MCPBridge(
        server_specs=[],
        tool_registry=tool_registry,
        resource_uris=["file:///a"],
    )
    bridge._clients = [mock_mcp_client]
    mock_mcp_client.read_resource.return_value = "content A"
    result = await bridge.get_resource_context()
    assert "content A" in result
    mock_mcp_client.read_resource.assert_called_with("file:///a")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_get_resource_context_uris_param(tool_registry, mock_mcp_client):
    """get_resource_context(uris=...) uses param over instance resource_uris."""
    bridge = MCPBridge(server_specs=[], tool_registry=tool_registry, resource_uris=["file:///default"])
    bridge._clients = [mock_mcp_client]
    mock_mcp_client.read_resource.return_value = "from param"
    result = await bridge.get_resource_context(uris=["file:///param"])
    mock_mcp_client.read_resource.assert_called_with("file:///param")
    assert "from param" in result
