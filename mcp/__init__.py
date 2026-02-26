"""
MCP (Model Context Protocol) integration for Curio Agent SDK.

Connect to MCP servers over stdio or HTTP and expose their tools, resources,
and prompts to the agent.

Quick (URL only):
    agent = Agent.builder() \\
        .model("openai:gpt-4o") \\
        .mcp_server("stdio://npx -y @modelcontextprotocol/server-filesystem /path") \\
        .build()

Cursor/Claude-style config (command, args, env for credentials; or url + headers):
    agent = Agent.builder() \\
        .model("openai:gpt-4o") \\
        .mcp_server_config({
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_TOKEN"},
        }) \\
        .build()

From JSON file (e.g. mcp.json):
    agent = Agent.builder().mcp_servers_from_file("mcp.json").build()

Env vars: use $VAR or ${VAR} in env/headers; they are resolved at connect time.
"""

from curio_agent_sdk.mcp.client import MCPClient
from curio_agent_sdk.mcp.adapter import MCPToolAdapter
from curio_agent_sdk.mcp.bridge import MCPBridge
from curio_agent_sdk.mcp.config import (
    MCPServerConfig,
    load_mcp_servers_from_file,
    resolve_env_in_config,
)

__all__ = [
    "MCPClient",
    "MCPToolAdapter",
    "MCPBridge",
    "MCPServerConfig",
    "load_mcp_servers_from_file",
    "resolve_env_in_config",
]
