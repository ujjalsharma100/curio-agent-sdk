"""
Integration tests: Plugin System (Phase 17 §21.12)

Validates plugin registration and agent integration.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.agent.builder import AgentBuilder
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.extensions.plugins import Plugin, apply_plugins_to_builder
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def plugin_tool(query: str) -> str:
    """A tool provided by a plugin."""
    return f"Plugin result: {query}"


@tool
def another_tool(x: str) -> str:
    """Another plugin tool."""
    return f"Another: {x}"


class MyPlugin(Plugin):
    """A test plugin that registers tools via builder."""

    name: str = "my_plugin"
    version: str = "1.0.0"

    def register(self, builder):
        builder.add_tool(plugin_tool)


class SecondPlugin(Plugin):
    """Another test plugin."""

    name: str = "second_plugin"
    version: str = "1.0.0"

    def register(self, builder):
        builder.add_tool(another_tool)


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_plugin_registration():
    """Plugin registers tools via builder.register()."""
    builder = AgentBuilder()
    builder.system_prompt("Test plugins.")

    plugin = MyPlugin(name="my_plugin", version="1.0.0")
    apply_plugins_to_builder(builder, [plugin])

    mock = MockLLM()
    builder.llm(mock)
    mock.add_text_response("Plugin loaded.")

    agent = builder.build()
    tool_names = agent.registry.names
    assert "plugin_tool" in tool_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_plugin_tool_execution():
    """Plugin tool can be called by the agent."""
    builder = AgentBuilder()
    builder.system_prompt("Use plugins.")

    plugin = MyPlugin(name="my_plugin", version="1.0.0")
    apply_plugins_to_builder(builder, [plugin])

    mock = MockLLM()
    mock.add_tool_call_response("plugin_tool", {"query": "test"})
    mock.add_text_response("Plugin tool executed.")
    builder.llm(mock)

    agent = builder.build()
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Use the plugin tool")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 1
    assert harness.tool_calls[0][0] == "plugin_tool"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_plugins():
    """Multiple plugins register their tools."""
    builder = AgentBuilder()
    builder.system_prompt("Multi-plugin.")

    plugins = [
        MyPlugin(name="my_plugin", version="1.0.0"),
        SecondPlugin(name="second_plugin", version="1.0.0"),
    ]
    apply_plugins_to_builder(builder, plugins)

    mock = MockLLM()
    mock.add_text_response("Both plugins loaded.")
    builder.llm(mock)

    agent = builder.build()
    tool_names = agent.registry.names
    assert "plugin_tool" in tool_names
    assert "another_tool" in tool_names
