"""
MCP server configuration: Cursor/Claude-style JSON with command, args, env, headers.

Supports both stdio servers (command + args + env for credentials) and
remote HTTP servers (url + headers for auth).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPServerConfig:
    """
    Configuration for a single MCP server.

    Use for stdio servers (command + args + env) or HTTP servers (url + headers).
    Matches Cursor/Claude mcp.json style.

    Stdio example:
        MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."},
        )

    HTTP example:
        MCPServerConfig(
            name="remote",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer ...", "X-API-Key": "..."},
            timeout=60.0,
        )
    """

    # Identity (optional, for logging / tool prefix)
    name: str = ""

    # Stdio server
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # HTTP server
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0

    # Common
    disabled: bool = False

    def is_stdio(self) -> bool:
        return bool(self.command)

    def is_http(self) -> bool:
        return bool(self.url and not self.command)

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str = "") -> MCPServerConfig:
        """
        Build from a single server config dict (Cursor/Claude style).

        Accepts:
            - command, args, env (stdio)
            - url, headers, timeout (HTTP)
            - disabled
        """
        name = name or data.get("name", "")
        env = data.get("env") or {}
        if not isinstance(env, dict):
            env = {}
        env = {str(k): str(v) for k, v in env.items()}
        headers = data.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        headers = {str(k): str(v) for k, v in headers.items()}
        args = data.get("args") or []
        if isinstance(args, list):
            args = [str(a) for a in args]
        else:
            args = []
        return cls(
            name=name,
            command=str(data.get("command", "")).strip(),
            args=args,
            env=env,
            url=str(data.get("url", "")).strip(),
            headers=headers,
            timeout=float(data.get("timeout", 30)),
            disabled=bool(data.get("disabled", False)),
        )


def load_mcp_servers_from_file(path: str | Path) -> list[MCPServerConfig]:
    """
    Load MCP server configs from a JSON file (Cursor/Claude mcp.json style).

    Expected format:
        {
          "mcpServers": {
            "server-name": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-github"],
              "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "..." }
            },
            "remote": {
              "url": "https://example.com/mcp",
              "headers": { "Authorization": "Bearer ..." }
            }
          }
        }

    Returns a list of MCPServerConfig (skips disabled). Use env vars for
    secrets in production (e.g. env value "$GITHUB_TOKEN" can be resolved
    via resolve_env_in_config).
    """
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    servers = data.get("mcpServers") or data.get("mcp_servers") or {}
    if isinstance(servers, list):
        out = []
        for i, s in enumerate(servers):
            if isinstance(s, dict):
                cfg = MCPServerConfig.from_dict(s, name=s.get("name", "") or f"server_{i}")
                if not cfg.disabled:
                    out.append(cfg)
        return out
    out = []
    for name, s in servers.items():
        if isinstance(s, dict):
            cfg = MCPServerConfig.from_dict(s, name=name)
            if not cfg.disabled:
                out.append(cfg)
    return out


def resolve_env_in_config(config: MCPServerConfig) -> MCPServerConfig:
    """
    Resolve environment variable references in env and headers.

    Values like "$VAR" or "${VAR}" are replaced with os.environ.get("VAR", "").
    Use this to avoid hardcoding secrets in config files.
    """
    def resolve(val: str) -> str:
        if not val or (not val.startswith("$") and "${" not in val):
            return val
        s = val.strip()
        if s.startswith("${") and s.endswith("}"):
            key = s[2:-1]
            return os.environ.get(key, "")
        if s.startswith("$"):
            key = s[1:]
            return os.environ.get(key, "")
        return val

    env = {k: resolve(v) for k, v in config.env.items()}
    headers = {k: resolve(v) for k, v in config.headers.items()}
    return MCPServerConfig(
        name=config.name,
        command=config.command,
        args=config.args,
        env=env,
        url=config.url,
        headers=headers,
        timeout=config.timeout,
        disabled=config.disabled,
    )
