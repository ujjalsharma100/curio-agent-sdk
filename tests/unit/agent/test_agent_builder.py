"""
Unit tests for curio_agent_sdk.core.agent.builder

Covers: AgentBuilder — fluent API for all configuration methods,
chaining, build, validation
"""

from unittest.mock import MagicMock

import pytest

from curio_agent_sdk.core.agent.builder import AgentBuilder
from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.core.extensions import Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_tool() -> Tool:
    """Create a minimal Tool for testing."""
    def _noop():
        return "result"
    return Tool(
        func=_noop,
        name="test_tool",
        description="A test tool",
    )


def _dummy_handler(ctx):
    pass


# ===================================================================
# Tests
# ===================================================================


class TestAgentBuilder:

    def test_builder_system_prompt(self):
        """.system_prompt() sets prompt."""
        b = AgentBuilder()
        result = b.system_prompt("Custom prompt")
        assert result is b  # fluent
        assert b._config["system_prompt"] == "Custom prompt"

    def test_builder_model(self):
        """.model() sets model."""
        b = AgentBuilder()
        b.model("openai:gpt-4o")
        assert b._config["model"] == "openai:gpt-4o"

    def test_builder_tier(self):
        """.tier() sets tier."""
        b = AgentBuilder()
        b.tier("tier1")
        assert b._config["tier"] == "tier1"

    def test_builder_llm(self):
        """.llm() sets client."""
        b = AgentBuilder()
        mock_llm = MagicMock()
        b.llm(mock_llm)
        assert b._config["llm"] is mock_llm

    def test_builder_loop(self):
        """.loop() sets loop."""
        b = AgentBuilder()
        mock_loop = MagicMock()
        b.loop(mock_loop)
        assert b._config["loop"] is mock_loop

    def test_builder_tools(self):
        """.tools() sets tool list."""
        b = AgentBuilder()
        tools = [_dummy_tool()]
        b.tools(tools)
        assert b._config["tools"] == tools

    def test_builder_tool_single(self):
        """.add_tool() adds one tool."""
        b = AgentBuilder()
        t = _dummy_tool()
        b.add_tool(t)
        assert t in b._config["tools"]

    def test_builder_max_iterations(self):
        """.max_iterations() sets limit."""
        b = AgentBuilder()
        b.max_iterations(50)
        assert b._config["max_iterations"] == 50

    def test_builder_timeout(self):
        """.timeout() sets timeout."""
        b = AgentBuilder()
        b.timeout(120.0)
        assert b._config["timeout"] == 120.0

    def test_builder_temperature(self):
        """.temperature() sets temp."""
        b = AgentBuilder()
        b.temperature(0.2)
        assert b._config["temperature"] == 0.2

    def test_builder_middleware(self):
        """.middleware() sets pipeline."""
        b = AgentBuilder()
        mw = [MagicMock()]
        b.middleware(mw)
        assert b._config["middleware"] == mw

    def test_builder_human_input(self):
        """.human_input() sets handler."""
        b = AgentBuilder()
        handler = MagicMock()
        b.human_input(handler)
        assert b._config["human_input"] is handler

    def test_builder_permission_policy(self):
        """.permissions() sets policy."""
        b = AgentBuilder()
        policy = MagicMock()
        b.permissions(policy)
        assert b._config["permission_policy"] is policy

    def test_builder_memory_manager(self):
        """.memory_manager() sets manager."""
        b = AgentBuilder()
        mm = MagicMock()
        b.memory_manager(mm)
        assert b._config["memory_manager"] is mm

    def test_builder_state_store(self):
        """.state_store() sets store."""
        b = AgentBuilder()
        store = MagicMock()
        b.state_store(store)
        assert b._config["state_store"] is store

    def test_builder_instructions(self):
        """.instructions() sets text."""
        b = AgentBuilder()
        b.instructions("Always respond in JSON.")
        assert b._config["instructions"] == "Always respond in JSON."

    def test_builder_instructions_file(self):
        """.instructions_file() sets path."""
        b = AgentBuilder()
        b.instructions_file("./AGENT.md")
        assert b._config["instructions_file"] == "./AGENT.md"

    def test_builder_hook(self):
        """.hook() registers handler."""
        b = AgentBuilder()
        b.hook("agent.run.before", _dummy_handler, priority=5)
        assert len(b._config["hooks"]) == 1
        event, handler, priority = b._config["hooks"][0]
        assert event == "agent.run.before"
        assert handler is _dummy_handler
        assert priority == 5

    def test_builder_skill(self):
        """.skill() adds skill."""
        b = AgentBuilder()
        s = Skill(name="test_skill", tools=[], description="Test")
        b.skill(s)
        assert s in b._config["skills"]

    def test_builder_subagent(self):
        """.subagent() adds config."""
        b = AgentBuilder()
        config = {"system_prompt": "Research helper", "tools": []}
        b.subagent("researcher", config)
        assert "researcher" in b._config["subagent_configs"]

    def test_builder_mcp_server(self):
        """.mcp_server() adds MCP."""
        b = AgentBuilder()
        b.mcp_server("http://localhost:8080/sse")
        assert "http://localhost:8080/sse" in b._config["mcp_server_urls"]

    def test_builder_connector(self):
        """.connector() adds connector."""
        b = AgentBuilder()
        conn = MagicMock()
        b.connector(conn)
        assert conn in b._config["connectors"]

    def test_builder_plugin(self):
        """.plugin() applies plugin."""
        b = AgentBuilder()
        plug = MagicMock()
        b.plugin(plug)
        assert plug in b._config["plugins"]

    def test_builder_chaining(self):
        """Fluent chaining returns self."""
        b = AgentBuilder()
        result = (
            b.system_prompt("Test")
            .model("openai:gpt-4o")
            .tier("tier1")
            .max_iterations(10)
            .timeout(30.0)
            .temperature(0.5)
        )
        assert result is b

    def test_builder_build(self):
        """.build() returns Agent."""
        from curio_agent_sdk.core.agent.agent import Agent

        agent = AgentBuilder().system_prompt("Build test").build()
        assert isinstance(agent, Agent)
        assert agent.system_prompt == "Build test"

    def test_builder_build_with_model(self):
        """.build() with model creates proper agent."""
        from curio_agent_sdk.core.agent.agent import Agent

        agent = (
            AgentBuilder()
            .model("openai:gpt-4o")
            .system_prompt("Model test")
            .max_iterations(5)
            .build()
        )
        assert isinstance(agent, Agent)
        assert agent.max_iterations == 5

    def test_builder_event_bus(self):
        """.event_bus() sets bus."""
        b = AgentBuilder()
        bus = MagicMock()
        b.event_bus(bus)
        assert b._config["event_bus"] is bus

    def test_builder_checkpoint_interval(self):
        """.checkpoint_interval() sets interval."""
        b = AgentBuilder()
        b.checkpoint_interval(5)
        assert b._config["checkpoint_interval"] == 5

    def test_builder_clone(self):
        """clone() creates independent copy."""
        b1 = AgentBuilder().system_prompt("Original")
        b2 = b1.clone()
        b2.system_prompt("Clone")
        assert b1._config["system_prompt"] == "Original"
        assert b2._config["system_prompt"] == "Clone"

    def test_builder_repr(self):
        """__repr__ shows configured keys."""
        b = AgentBuilder().model("openai:gpt-4o").tier("tier1")
        r = repr(b)
        assert "model" in r
        assert "tier" in r
