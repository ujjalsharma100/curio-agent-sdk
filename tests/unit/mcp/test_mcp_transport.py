"""
Unit tests for MCP transport — MCPTransport ABC, MCPError, transport_for_url,
transport_from_config, StdioTransport (construction only), HTTPTransport connect/request.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from curio_agent_sdk.mcp.transport import (
    MCPTransport,
    MCPError,
    transport_for_url,
    transport_from_config,
    StdioTransport,
    HTTPTransport,
)
from curio_agent_sdk.mcp.config import MCPServerConfig


# ---------------------------------------------------------------------------
# MCPError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_error_creation():
    """MCPError stores message, code, data."""
    err = MCPError("Something failed", code=-32600, data={"detail": "x"})
    assert str(err) == "Something failed"
    assert err.code == -32600
    assert err.data == {"detail": "x"}


@pytest.mark.unit
def test_mcp_error_defaults():
    """MCPError code and data can be None."""
    err = MCPError("msg")
    assert err.code is None
    assert err.data is None


# ---------------------------------------------------------------------------
# transport_for_url
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_transport_for_url_stdio():
    """transport_for_url with stdio:// returns StdioTransport."""
    t = transport_for_url("stdio://npx -y @mcp/server")
    assert isinstance(t, StdioTransport)


@pytest.mark.unit
def test_transport_for_url_http():
    """transport_for_url with http:// returns HTTPTransport."""
    t = transport_for_url("http://localhost:8080")
    assert isinstance(t, HTTPTransport)


@pytest.mark.unit
def test_transport_for_url_https():
    """transport_for_url with https:// returns HTTPTransport."""
    t = transport_for_url("https://mcp.example.com", timeout=60.0)
    assert isinstance(t, HTTPTransport)
    assert t._timeout == 60.0


@pytest.mark.unit
def test_transport_for_url_unsupported():
    """transport_for_url with unsupported scheme raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        transport_for_url("ftp://example.com")


# ---------------------------------------------------------------------------
# transport_from_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_transport_from_config_stdio():
    """transport_from_config with command returns StdioTransport."""
    cfg = MCPServerConfig(name="x", command="npx", args=["-y", "server"])
    t = transport_from_config(cfg)
    assert isinstance(t, StdioTransport)


@pytest.mark.unit
def test_transport_from_config_http():
    """transport_from_config with url returns HTTPTransport."""
    cfg = MCPServerConfig(name="x", url="https://mcp.example.com", headers={"Auth": "x"})
    t = transport_from_config(cfg)
    assert isinstance(t, HTTPTransport)
    assert t._server_url == "https://mcp.example.com"
    assert t._headers.get("Auth") == "x"


@pytest.mark.unit
def test_transport_from_config_neither_raises():
    """transport_from_config with no command and no url raises ValueError."""
    cfg = MCPServerConfig(name="x")  # no command, no url
    with pytest.raises(ValueError, match="command.*url"):
        transport_from_config(cfg)


# ---------------------------------------------------------------------------
# StdioTransport (construction only; no real subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stdio_transport_from_url():
    """StdioTransport from stdio:// URL parses command string."""
    t = StdioTransport(server_url="stdio://npx -y server")
    assert t._server_url == "stdio://npx -y server"
    assert t._command_str == "npx -y server"


@pytest.mark.unit
def test_stdio_transport_from_url_invalid():
    """StdioTransport with non-stdio URL raises ValueError."""
    with pytest.raises(ValueError, match="stdio://"):
        StdioTransport(server_url="http://x")


@pytest.mark.unit
def test_stdio_transport_from_command_args():
    """StdioTransport from command/args."""
    t = StdioTransport(command="npx", args=["-y", "server"], env={"K": "v"})
    assert t._command == "npx"
    assert t._args == ["-y", "server"]
    assert t._env == {"K": "v"}


@pytest.mark.unit
def test_stdio_transport_from_command_required():
    """StdioTransport without server_url requires command."""
    with pytest.raises(ValueError, match="command"):
        StdioTransport(server_url=None)


# ---------------------------------------------------------------------------
# HTTPTransport
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_transport_connect():
    """HTTPTransport connect sets _connected."""
    t = HTTPTransport("http://localhost:8080")
    assert t._connected is False
    await t.connect()
    assert t._connected is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_transport_disconnect():
    """HTTPTransport disconnect clears _connected."""
    t = HTTPTransport("http://localhost:8080")
    t._connected = True
    await t.disconnect()
    assert t._connected is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_transport_request_not_connected():
    """HTTPTransport request when not connected raises MCPError."""
    t = HTTPTransport("http://localhost:8080")
    with pytest.raises(MCPError, match="Not connected"):
        await t.request("tools/list", None)


@pytest.mark.unit
def test_http_transport_strips_trailing_slash():
    """HTTPTransport strips trailing slash from URL."""
    t = HTTPTransport("https://api.example.com/mcp/")
    assert t._server_url == "https://api.example.com/mcp"
