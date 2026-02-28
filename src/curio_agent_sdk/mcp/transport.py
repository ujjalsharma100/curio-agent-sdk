"""
MCP transports: stdio (subprocess) and HTTP.

Uses only Python stdlib. Messages are JSON-RPC 2.0; stdio uses newline-delimited JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract transport for MCP JSON-RPC communication."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        ...

    @abstractmethod
    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Send a JSON-RPC request and return the result.

        Raises:
            MCPError: On protocol or application error.
        """
        ...


class MCPError(Exception):
    """MCP protocol or server error."""

    def __init__(self, message: str, code: int | None = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def _next_id() -> int:
    """Simple monotonic request id (module-level counter)."""
    if not hasattr(_next_id, "_counter"):
        _next_id._counter = 0
    _next_id._counter += 1
    return _next_id._counter


def _encode_message(msg: dict[str, Any]) -> bytes:
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def _decode_message(line: bytes) -> dict[str, Any]:
    return json.loads(line.decode("utf-8").strip())


class StdioTransport(MCPTransport):
    """
    MCP transport over stdio: spawns a subprocess and communicates via stdin/stdout.

    Can be constructed from:
    - server_url: stdio://<command> [args...]  (env not supported in URL)
    - command + args + optional env: for Cursor/Claude-style config (credentials via env)
    """

    def __init__(
        self,
        server_url: str | None = None,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        if server_url is not None:
            if not server_url.startswith("stdio://"):
                raise ValueError("Stdio URL must start with stdio://")
            self._server_url = server_url
            self._command_str = server_url[len("stdio://") :].strip()
            self._command: str | None = None
            self._args: list[str] = []
            self._env: dict[str, str] = {}
        else:
            if not command:
                raise ValueError("Either server_url or command must be provided")
            self._server_url = None
            self._command_str = None
            self._command = command
            self._args = list(args or [])
            self._env = dict(env or [])
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._command_str is not None:
            parts = shlex.split(self._command_str)
            if not parts:
                raise ValueError("Stdio command is empty")
            cmd, args = parts[0], parts[1:]
            env = None
        else:
            cmd = self._command
            args = self._args
            # Merge env with current process env so server gets credentials
            if self._env:
                env = {**os.environ, **self._env}
            else:
                env = None
        self._process = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        self._reader = self._process.stdout
        self._writer = self._process.stdin
        logger.debug("MCP stdio process started: %s %s", cmd, args)
        if env and self._env:
            logger.debug("MCP stdio env keys set: %s", list(self._env.keys()))

    async def disconnect(self) -> None:
        if self._process is None:
            return
        try:
            if self._writer and not self._writer.is_closing():
                self._writer.close()
                await self._writer.wait_closed()
        except Exception as e:
            logger.debug("Error closing MCP stdio writer: %s", e)
        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()
        except Exception as e:
            logger.debug("Error terminating MCP stdio process: %s", e)
        self._process = None
        self._reader = None
        self._writer = None

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._reader is None or self._writer is None:
            raise MCPError("Not connected")
        req = {
            "jsonrpc": "2.0",
            "id": _next_id(),
            "method": method,
            "params": params if params is not None else {},
        }
        async with self._lock:
            self._writer.write(_encode_message(req))
            await self._writer.drain()
            line = await self._reader.readline()
        if not line:
            raise MCPError("Connection closed by server")
        resp = _decode_message(line)
        if "error" in resp:
            err = resp["error"]
            raise MCPError(
                err.get("message", "Unknown error"),
                code=err.get("code"),
                data=err.get("data"),
            )
        return resp.get("result", {})


class HTTPTransport(MCPTransport):
    """
    MCP transport over HTTP: POST JSON-RPC to a URL, read JSON response.

    Supports optional headers (e.g. Authorization, X-API-Key) for remote servers.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ):
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self._headers = dict(headers or [])
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._connected:
            raise MCPError("Not connected")
        import urllib.request
        import urllib.error

        req_body = json.dumps({
            "jsonrpc": "2.0",
            "id": _next_id(),
            "method": method,
            "params": params if params is not None else {},
        }).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **self._headers}
        req = urllib.request.Request(
            self._server_url,
            data=req_body,
            headers=req_headers,
            method="POST",
        )
        try:
            def _do_request():
                with urllib.request.urlopen(req, timeout=self._timeout) as f:
                    return json.loads(f.read().decode("utf-8"))
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, _do_request),
                timeout=self._timeout + 5.0,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            try:
                err_data = json.loads(body)
                err = err_data.get("error", {})
                raise MCPError(
                    err.get("message", str(e)),
                    code=err.get("code", e.code),
                    data=err.get("data"),
                )
            except (json.JSONDecodeError, MCPError):
                raise MCPError(f"HTTP {e.code}: {body or str(e)}")
        except asyncio.TimeoutError:
            raise MCPError("HTTP request timed out")
        if "error" in resp:
            err = resp["error"]
            raise MCPError(
                err.get("message", "Unknown error"),
                code=err.get("code"),
                data=err.get("data"),
            )
        return resp.get("result", {})


def transport_for_url(server_url: str, **kwargs: Any) -> MCPTransport:
    """Create the appropriate transport for a server URL."""
    if server_url.startswith("stdio://"):
        return StdioTransport(server_url)
    if server_url.startswith("http://") or server_url.startswith("https://"):
        return HTTPTransport(
            server_url,
            timeout=kwargs.get("timeout", 30.0),
            headers=kwargs.get("headers"),
        )
    raise ValueError(f"Unsupported MCP server URL scheme: {server_url.split('://')[0] if '://' in server_url else server_url}")


def transport_from_config(config: Any) -> MCPTransport:
    """
    Create a transport from an MCPServerConfig (Cursor/Claude-style).

    Stdio: command + args + env. HTTP: url + headers + timeout.
    """
    from curio_agent_sdk.mcp.config import MCPServerConfig

    if not isinstance(config, MCPServerConfig):
        config = MCPServerConfig.from_dict(config if isinstance(config, dict) else {}, name="")
    if config.is_stdio():
        return StdioTransport(
            server_url=None,
            command=config.command,
            args=config.args,
            env=config.env,
        )
    if config.is_http():
        return HTTPTransport(
            config.url,
            timeout=config.timeout,
            headers=config.headers or None,
        )
    raise ValueError("Config must have command (stdio) or url (HTTP)")
