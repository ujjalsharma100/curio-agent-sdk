"""
Unit tests for Connector base class — abstract, concrete impl, connect/disconnect,
get_tools, health_check, ConnectorResource, resolve_credentials.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from curio_agent_sdk.connectors.base import (
    Connector,
    ConnectorResource,
    resolve_credentials,
)
from curio_agent_sdk.core.tools.tool import Tool


# ---------------------------------------------------------------------------
# Connector is abstract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_connector_is_abstract():
    """Connector cannot be instantiated directly (abstract)."""
    with pytest.raises(TypeError):
        Connector()


# ---------------------------------------------------------------------------
# Concrete connector implementation
# ---------------------------------------------------------------------------


class ConcreteConnector(Connector):
    """Minimal concrete Connector for testing."""
    name = "test_connector"
    _tools: list

    def __init__(self):
        self._tools = []
        self._connected = False

    async def connect(self, credentials: dict | None = None) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def add_tool(self, tool: Tool) -> None:
        self._tools.append(tool)


@pytest.mark.unit
def test_connector_concrete_impl():
    """Concrete connector can be instantiated and has required interface."""
    c = ConcreteConnector()
    assert c.name == "test_connector"
    assert hasattr(c, "connect")
    assert hasattr(c, "disconnect")
    assert hasattr(c, "get_tools")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connector_connect_disconnect():
    """Concrete connector connect/disconnect lifecycle."""
    c = ConcreteConnector()
    await c.connect()
    assert c._connected is True
    await c.disconnect()
    assert c._connected is False


@pytest.mark.unit
def test_connector_get_tools():
    """get_tools returns tools from connector."""
    c = ConcreteConnector()
    tool = Tool(func=lambda: None, name="my_tool", description="Desc")
    c.add_tool(tool)
    tools = c.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "my_tool"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connector_health_check():
    """health_check default returns True."""
    c = ConcreteConnector()
    assert await c.health_check() is True


@pytest.mark.unit
def test_connector_get_resources_default():
    """get_resources default returns empty list."""
    c = ConcreteConnector()
    assert c.get_resources() == []


# ---------------------------------------------------------------------------
# ConnectorResource
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_connector_resource():
    """ConnectorResource has uri, content, mime_type."""
    r = ConnectorResource(uri="doc://api", content="API docs here", mime_type="text/plain")
    assert r.uri == "doc://api"
    assert r.content == "API docs here"
    assert r.mime_type == "text/plain"


# ---------------------------------------------------------------------------
# resolve_credentials
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_credentials_no_refs():
    """resolve_credentials leaves dict without $ refs unchanged."""
    out = resolve_credentials({"api_key": "plain"})
    assert out == {"api_key": "plain"}


@pytest.mark.unit
def test_resolve_credentials_env_ref(monkeypatch):
    """resolve_credentials expands $VAR from environment."""
    monkeypatch.setenv("TEST_CONNECTOR_KEY", "secret")
    try:
        out = resolve_credentials({"api_key": "$TEST_CONNECTOR_KEY"})
        assert out["api_key"] == "secret"
    finally:
        monkeypatch.delenv("TEST_CONNECTOR_KEY", raising=False)
