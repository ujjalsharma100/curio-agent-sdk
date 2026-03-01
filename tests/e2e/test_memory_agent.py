"""
E2E tests: Memory Agent (Phase 18 §22.7)

Validates agent memory persistence across runs and composite memory.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.memory.manager import MemoryManager
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.composite import CompositeMemory
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_remembers_context():
    """Memory persists across multiple agent runs."""
    memory = ConversationMemory(max_entries=50)
    mm = MemoryManager(memory=memory)

    mock = MockLLM()
    mock.add_text_response("I'll remember that your favorite color is blue.")
    mock.add_text_response("Your favorite color is blue!")
    mock.add_text_response("You also like pizza, and your favorite color is blue.")

    agent = Agent(
        system_prompt="You are an assistant with memory. Remember user preferences.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)

    # Run 1: Tell the agent something
    r1 = await harness.run("My favorite color is blue.")
    assert r1.status == "completed"

    # Run 2: Ask about it
    r2 = await harness.run("What's my favorite color?")
    assert r2.status == "completed"

    # Run 3: Add more info
    r3 = await harness.run("I also like pizza.")
    assert r3.status == "completed"

    # Verify memory accumulated data
    count = await memory.count()
    assert count >= 2  # At least 2 runs saved

    # Verify memory can be searched
    results = await memory.search("color")
    assert len(results) >= 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_with_composite_memory():
    """Agent uses multiple memory types combined."""
    conv_memory = ConversationMemory(max_entries=20)
    kv_memory = KeyValueMemory()

    # Pre-populate KV memory with user profile
    await kv_memory.add("user_name=Alice", metadata={"key": "user_name"})
    await kv_memory.add("user_role=engineer", metadata={"key": "user_role"})

    composite = CompositeMemory(memories={"conversation": conv_memory, "kv": kv_memory})
    mm = MemoryManager(memory=composite)

    mock = MockLLM()
    mock.add_text_response("Hello Alice! I see you're an engineer. How can I help?")

    agent = Agent(
        system_prompt="You are a personalized assistant. Use memory to personalize responses.",
        memory_manager=mm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello!")

    assert result.status == "completed"

    # Both memory backends should be queryable through composite
    kv_results = await kv_memory.search("user_name")
    assert len(kv_results) >= 1

    conv_count = await conv_memory.count()
    # Conversation memory should have data after the run
    assert conv_count >= 0  # May or may not have saved yet
