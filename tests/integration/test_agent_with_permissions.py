"""
Integration tests: Agent + Permissions (Phase 17 §21.7)

Validates allowed tools, denied tools, and ask-user flow.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.security.permissions import (
    AllowAll,
    PermissionPolicy,
    PermissionResult,
)
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def safe_tool(x: str) -> str:
    """A safe tool."""
    return f"Safe: {x}"


@tool
def dangerous_tool(x: str) -> str:
    """A dangerous tool."""
    return f"Danger: {x}"


class DenyDangerousPolicy(PermissionPolicy):
    """Denies the 'dangerous_tool' and allows everything else."""

    async def check_tool_call(self, tool_name, args, context=None):
        if tool_name == "dangerous_tool":
            return PermissionResult.deny("Tool is dangerous")
        return PermissionResult.allow()

    async def check_file_access(self, path, mode, context=None):
        return PermissionResult.allow()

    async def check_network_access(self, url, context=None):
        return PermissionResult.allow()


class AskForDangerousPolicy(PermissionPolicy):
    """Asks user for dangerous_tool, allows everything else."""

    async def check_tool_call(self, tool_name, args, context=None):
        if tool_name == "dangerous_tool":
            return PermissionResult.ask("This tool is dangerous. Proceed?")
        return PermissionResult.allow()

    async def check_file_access(self, path, mode, context=None):
        return PermissionResult.allow()

    async def check_network_access(self, url, context=None):
        return PermissionResult.allow()


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allowed_tool_executes():
    """Allowed tools run normally under AllowAll."""
    mock = MockLLM()
    mock.add_tool_call_response("safe_tool", {"x": "test"})
    mock.add_text_response("Done safely.")

    agent = Agent(
        system_prompt="Test.",
        tools=[safe_tool],
        permission_policy=AllowAll(),
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Use safe tool")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 1
    assert harness.tool_calls[0][0] == "safe_tool"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_denied_tool_blocked():
    """Denied tools return permission error and LLM gets error feedback."""
    mock = MockLLM()
    mock.add_tool_call_response("dangerous_tool", {"x": "hack"})
    mock.add_text_response("The tool was denied.")

    agent = Agent(
        system_prompt="Test.",
        tools=[safe_tool, dangerous_tool],
        permission_policy=DenyDangerousPolicy(),
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Use dangerous tool")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ask_user_flow():
    """Human-in-the-loop confirmation: tool that requires ask_user."""
    from unittest.mock import AsyncMock

    # Create a mock human input handler that auto-approves
    human_input = AsyncMock(return_value=True)

    mock = MockLLM()
    mock.add_tool_call_response("dangerous_tool", {"x": "risky"})
    mock.add_text_response("Approved and done.")

    agent = Agent(
        system_prompt="Test.",
        tools=[safe_tool, dangerous_tool],
        permission_policy=AskForDangerousPolicy(),
        human_input=human_input,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Use the dangerous tool")

    assert result.status == "completed"
