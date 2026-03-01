"""
Integration tests: Memory + Persistence (Phase 17 §21.12)

Validates memory with persistence backend.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.memory.manager import MemoryManager
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_memory_persists_across_runs():
    """Conversation memory retains data between agent runs."""
    memory = ConversationMemory(max_entries=20)
    mm = MemoryManager(memory=memory)

    mock = MockLLM()
    mock.add_text_response("First answer.")
    mock.add_text_response("Recalled from memory.")

    agent = Agent(
        system_prompt="Remember everything.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)

    await harness.run("Remember this: important fact.")
    count_after_first = await memory.count()
    assert count_after_first > 0

    await harness.run("What did I tell you?")
    count_after_second = await memory.count()
    assert count_after_second >= count_after_first


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kv_memory_stores_and_retrieves():
    """KeyValueMemory stores and retrieves structured data."""
    memory = KeyValueMemory()
    mm = MemoryManager(memory=memory)

    # Pre-populate memory
    await memory.add("project_name=CurioSDK", metadata={"key": "project_name"})

    mock = MockLLM()
    mock.add_text_response("Found: CurioSDK.")

    agent = Agent(
        system_prompt="Use memory.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("What is the project name?")

    assert result.status == "completed"

    # Verify memory has data
    entries = await memory.search("project_name")
    assert len(entries) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_clear_resets():
    """Clearing memory removes all entries."""
    memory = ConversationMemory(max_entries=10)
    mm = MemoryManager(memory=memory)

    await memory.add("Data to clear.")
    assert await memory.count() > 0

    await memory.clear()
    assert await memory.count() == 0

    mock = MockLLM()
    mock.add_text_response("Starting fresh.")

    agent = Agent(
        system_prompt="Test.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello")

    assert result.status == "completed"
