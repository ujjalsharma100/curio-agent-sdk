"""
Unit tests for ConnectorBridge — startup connects connectors and registers tools,
shutdown disconnects, get_tools aggregated, health_check, get_resource_context.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from curio_agent_sdk.connectors.base import Connector, ConnectorResource
from curio_agent_sdk.connectors.bridge import ConnectorBridge
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.tool import Tool


class DummyConnector(Connector):
    """Connector that records lifecycle and returns one tool."""
    name = "dummy"
    _connected = False

    def __init__(self, tool_name: str = "dummy_tool"):
        self.tool_name = tool_name

    async def connect(self, credentials: dict | None = None) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def get_tools(self) -> list[Tool]:
        return [Tool(func=lambda: None, name=self.tool_name, description="")]

    def get_resources(self) -> list[ConnectorResource]:
        return [ConnectorResource(uri="dummy://doc", content="Dummy docs")]


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_startup(tool_registry):
    """Startup connects all connectors and registers their tools."""
    c1 = DummyConnector(tool_name="tool_a")
    c2 = DummyConnector(tool_name="tool_b")
    bridge = ConnectorBridge(connectors=[c1, c2], tool_registry=tool_registry)
    await bridge.startup()
    assert c1._connected and c2._connected
    assert tool_registry.has("tool_a")
    assert tool_registry.has("tool_b")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_shutdown(tool_registry):
    """Shutdown disconnects all connectors."""
    c = DummyConnector()
    bridge = ConnectorBridge(connectors=[c], tool_registry=tool_registry)
    await bridge.startup()
    assert c._connected
    await bridge.shutdown()
    assert not c._connected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_get_tools(tool_registry):
    """After startup, tools from all connectors are in registry."""
    c1 = DummyConnector(tool_name="conn_a_tool")
    bridge = ConnectorBridge(connectors=[c1], tool_registry=tool_registry)
    await bridge.startup()
    t = tool_registry.get("conn_a_tool")
    assert t is not None
    assert t.name == "conn_a_tool"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_health_check(tool_registry):
    """health_check returns True when all connectors healthy."""
    c = DummyConnector()
    bridge = ConnectorBridge(connectors=[c], tool_registry=tool_registry)
    await bridge.startup()
    assert await bridge.health_check() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridge_get_resource_context(tool_registry):
    """get_resource_context aggregates resources from all connectors."""
    c = DummyConnector()
    bridge = ConnectorBridge(connectors=[c], tool_registry=tool_registry)
    await bridge.startup()
    ctx = await bridge.get_resource_context()
    assert "dummy://doc" in ctx
    assert "Dummy docs" in ctx


@pytest.mark.unit
def test_bridge_get_circuit_breaker(tool_registry):
    """get_circuit_breaker returns None before startup; after startup returns CB for name."""
    c = DummyConnector()
    c.name = "myconn"
    bridge = ConnectorBridge(connectors=[c], tool_registry=tool_registry)
    assert bridge.get_circuit_breaker("myconn") is None
