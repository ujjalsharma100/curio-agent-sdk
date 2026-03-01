"""
Unit tests for MCPServerConfig and config helpers — creation, from_dict,
load_mcp_servers_from_file, resolve_env_in_config.
"""

import json
import os
from pathlib import Path

import pytest

from curio_agent_sdk.mcp.config import (
    MCPServerConfig,
    load_mcp_servers_from_file,
    resolve_env_in_config,
)


@pytest.mark.unit
class TestMCPServerConfigCreation:
    def test_config_creation_stdio(self):
        """MCPServerConfig with command/args/env (stdio)."""
        cfg = MCPServerConfig(
            name="fs",
            command="npx",
            args=["-y", "@mcp/server-filesystem"],
            env={"API_KEY": "secret"},
        )
        assert cfg.name == "fs"
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@mcp/server-filesystem"]
        assert cfg.env == {"API_KEY": "secret"}
        assert cfg.is_stdio() is True
        assert cfg.is_http() is False

    def test_config_creation_http(self):
        """MCPServerConfig with url/headers (HTTP)."""
        cfg = MCPServerConfig(
            name="remote",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer x"},
            timeout=60.0,
        )
        assert cfg.url == "https://api.example.com/mcp"
        assert cfg.headers == {"Authorization": "Bearer x"}
        assert cfg.timeout == 60.0
        assert cfg.is_http() is True
        assert cfg.is_stdio() is False

    def test_config_from_dict_stdio(self):
        """from_dict builds stdio config from Cursor-style dict."""
        data = {
            "command": "npx",
            "args": ["-y", "@mcp/server-github"],
            "env": {"GITHUB_TOKEN": "ghp_xxx"},
        }
        cfg = MCPServerConfig.from_dict(data, name="github")
        assert cfg.name == "github"
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@mcp/server-github"]
        assert cfg.env == {"GITHUB_TOKEN": "ghp_xxx"}

    def test_config_from_dict_http(self):
        """from_dict builds HTTP config."""
        data = {
            "url": "https://mcp.example.com",
            "headers": {"X-API-Key": "key"},
            "timeout": 45,
        }
        cfg = MCPServerConfig.from_dict(data)
        assert cfg.url == "https://mcp.example.com"
        assert cfg.headers == {"X-API-Key": "key"}
        assert cfg.timeout == 45.0

    def test_config_from_dict_disabled(self):
        """from_dict respects disabled flag."""
        cfg = MCPServerConfig.from_dict({"command": "npx", "args": [], "disabled": True})
        assert cfg.disabled is True

    def test_config_from_dict_name_from_data(self):
        """from_dict uses name from data when name arg empty."""
        cfg = MCPServerConfig.from_dict({"name": "myserver", "command": "npx", "args": []})
        assert cfg.name == "myserver"


@pytest.mark.unit
class TestResolveEnvInConfig:
    def test_resolve_env_no_placeholders(self):
        """resolve_env_in_config leaves values without $ unchanged."""
        cfg = MCPServerConfig(name="x", command="npx", args=[], env={"K": "plain"})
        out = resolve_env_in_config(cfg)
        assert out.env == {"K": "plain"}

    def test_resolve_env_simple_var(self):
        """$VAR is replaced with os.environ[VAR]."""
        cfg = MCPServerConfig(name="x", command="npx", args=[], env={"TOKEN": "$TEST_MCP_TOKEN"})
        os.environ["TEST_MCP_TOKEN"] = "resolved"
        try:
            out = resolve_env_in_config(cfg)
            assert out.env["TOKEN"] == "resolved"
        finally:
            os.environ.pop("TEST_MCP_TOKEN", None)

    def test_resolve_env_braces_var(self):
        """${VAR} is replaced with os.environ[VAR]."""
        cfg = MCPServerConfig(name="x", command="npx", args=[], env={"K": "${TEST_MCP_BRACE}"})
        os.environ["TEST_MCP_BRACE"] = "value"
        try:
            out = resolve_env_in_config(cfg)
            assert out.env["K"] == "value"
        finally:
            os.environ.pop("TEST_MCP_BRACE", None)

    def test_resolve_env_missing_var(self):
        """Missing env var resolves to empty string."""
        cfg = MCPServerConfig(name="x", command="npx", args=[], env={"K": "$NONEXISTENT_VAR_123"})
        out = resolve_env_in_config(cfg)
        assert out.env["K"] == ""

    def test_resolve_env_in_headers(self):
        """Headers are also resolved."""
        cfg = MCPServerConfig(
            name="x",
            url="https://x.com",
            headers={"Authorization": "$TEST_MCP_AUTH"},
        )
        os.environ["TEST_MCP_AUTH"] = "Bearer xyz"
        try:
            out = resolve_env_in_config(cfg)
            assert out.headers["Authorization"] == "Bearer xyz"
        finally:
            os.environ.pop("TEST_MCP_AUTH", None)


@pytest.mark.unit
class TestLoadMcpServersFromFile:
    def test_load_file_not_exists(self, tmp_path):
        """Missing file returns empty list."""
        assert load_mcp_servers_from_file(tmp_path / "nonexistent.json") == []

    def test_load_file_mcp_servers_dict(self, tmp_path):
        """Load from mcpServers dict (Cursor style)."""
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({
            "mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "@mcp/server-fs"]},
            },
        }))
        configs = load_mcp_servers_from_file(path)
        assert len(configs) == 1
        assert configs[0].name == "fs"
        assert configs[0].command == "npx"

    def test_load_file_mcp_servers_list(self, tmp_path):
        """Load from mcp_servers list format."""
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({
            "mcp_servers": [
                {"name": "s1", "command": "npx", "args": []},
            ],
        }))
        configs = load_mcp_servers_from_file(path)
        assert len(configs) == 1
        assert configs[0].name == "s1"

    def test_load_file_skips_disabled(self, tmp_path):
        """Disabled servers are not returned."""
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({
            "mcpServers": {
                "on": {"command": "npx", "args": []},
                "off": {"command": "npx", "args": [], "disabled": True},
            },
        }))
        configs = load_mcp_servers_from_file(path)
        assert len(configs) == 1
        assert configs[0].name == "on"
