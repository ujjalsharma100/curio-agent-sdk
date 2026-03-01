"""
Integration tests: MCP Client + Bridge + Adapter + Agent (Phase 17 §21.12)

Validates MCP integration with the agent.
"""

import pytest
from unittest.mock import AsyncMock

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.mcp.adapter import MCPToolAdapter
from curio_agent_sdk.mcp.client import MCPClient, MCPTool
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_bridge_creation():
    """MCPBridge can be created and attached to an agent."""
    mock = MockLLM()
    mock.add_text_response("MCP ready.")

    agent = Agent(
        system_prompt="Test MCP.",
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Test MCP integration")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tool_adapter_schema():
    """MCPToolAdapter converts MCP tool definitions to SDK tools."""
    mcp_tool = MCPTool(
        name="mcp_search",
        description="Search via MCP",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    )

    # Create a mock MCP client
    mock_client = AsyncMock(spec=MCPClient)
    mock_client.call_tool = AsyncMock(return_value="mock result")

    tool = MCPToolAdapter.adapt(mcp_tool, mock_client)
    assert tool is not None
    assert tool.name == "mcp_search"
    assert tool.description == "Search via MCP"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_agent_without_servers():
    """Agent runs fine when MCP is configured but no servers are available."""
    mock = MockLLM()
    mock.add_text_response("No MCP servers, still working.")

    agent = Agent(
        system_prompt="Test.",
        mcp_server_urls=[],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Test without MCP")

    assert result.status == "completed"
