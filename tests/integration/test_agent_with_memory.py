"""
Integration tests: Agent + Memory (Phase 17 §21.2)

Validates memory injection, saving, cross-run persistence, and memory tools.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.memory.manager import MemoryManager
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_injected_into_context():
    """Memory content is added to messages before LLM call."""
    memory = ConversationMemory(max_entries=10)
    await memory.add("The user's name is Alice.")
    mm = MemoryManager(memory=memory)

    mock = MockLLM()
    mock.add_text_response("Hello Alice!")

    agent = Agent(
        system_prompt="Greet user by name.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hi there!")

    assert result.status == "completed"
    # The mock LLM received a call — check that memory-injected content was present
    assert mock.call_count >= 1
    llm_request = mock.calls[0]
    all_content = " ".join(
        getattr(m, "content", "") or "" for m in llm_request.messages
    )
    assert "Alice" in all_content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_saved_after_run():
    """Conversation is saved to memory after run completes."""
    memory = ConversationMemory(max_entries=10)
    mm = MemoryManager(memory=memory)

    mock = MockLLM()
    mock.add_text_response("I helped you with that.")

    agent = Agent(
        system_prompt="Helpful.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    await harness.run("Help me with something.")

    count = await memory.count()
    assert count > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_across_runs():
    """Second run uses memory from first run."""
    memory = ConversationMemory(max_entries=10)
    mm = MemoryManager(memory=memory)

    mock = MockLLM()
    mock.add_text_response("I'll remember your name is Bob.")
    mock.add_text_response("Hello Bob!")

    agent = Agent(
        system_prompt="Remember names.",
        memory_manager=mm,
        llm=mock,
    )

    # First run
    harness = AgentTestHarness(agent, llm=mock)
    await harness.run("My name is Bob.")

    # Second run — memory should have first conversation
    result2 = await harness.run("What's my name?")
    assert result2.status == "completed"
    assert mock.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_tools_available():
    """Memory management tools are provided by the memory manager."""
    memory = ConversationMemory(max_entries=10)
    mm = MemoryManager(memory=memory)

    # MemoryManager.get_tools() returns memory management tools
    memory_tools = mm.get_tools()
    tool_names = {t.name for t in memory_tools}
    expected = {"save_to_memory", "search_memory", "forget_memory"}
    assert expected.issubset(tool_names), (
        f"Expected memory tools {expected} in {tool_names}"
    )

    # When passed as tools to the agent, they work
    mock = MockLLM()
    mock.add_tool_call_response("search_memory", {"query": "test"})
    mock.add_text_response("Found in memory.")

    agent = Agent(
        system_prompt="Use memory tools.",
        memory_manager=mm,
        tools=memory_tools,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Search memory for test")

    assert result.status == "completed"
    assert len(harness.tool_calls) >= 1
