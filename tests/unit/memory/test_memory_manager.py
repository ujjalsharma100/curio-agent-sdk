"""
Unit tests for curio_agent_sdk.memory.manager — MemoryManager and strategies.
"""

import pytest

from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.memory import (
    ConversationMemory,
    MemoryManager,
    DefaultInjection,
    UserMessageInjection,
    NoInjection,
    DefaultSave,
    SaveEverythingStrategy,
    NoSave,
    PerIterationSave,
    DefaultQuery,
    KeywordQuery,
    AdaptiveTokenQuery,
)
from curio_agent_sdk.models.llm import Message


@pytest.fixture
def state():
    s = AgentState()
    s.add_message(Message.system("You are helpful."))
    s.add_message(Message.user("Hello"))
    return s


@pytest.fixture
def memory():
    return ConversationMemory(max_entries=20)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryManager:
    async def test_manager_inject_memory(self, memory, state):
        manager = MemoryManager(memory=memory, injection_strategy=DefaultInjection(max_tokens=500))
        await memory.add("Stored fact: user likes Python")
        await manager.inject(state, "What do I like?")
        # Should have inserted a system message with memory context
        assert len(state.messages) >= 2
        system_msgs = [m for m in state.messages if m.role == "system"]
        assert any("Python" in (m.text or "") for m in system_msgs)

    async def test_manager_save_memory(self, memory, state):
        manager = MemoryManager(memory=memory, save_strategy=DefaultSave())
        state.add_message(Message.assistant("I can help!"))
        await manager.on_run_end("Hello", "I can help!", state)
        assert await memory.count() >= 2  # user + assistant

    async def test_manager_on_iteration_end(self, memory, state):
        manager = MemoryManager(memory=memory, save_strategy=PerIterationSave())
        state.add_message(Message.assistant("Step 1 done"))
        await manager.on_iteration(state, 1)
        # PerIterationSave saves iteration snapshot
        assert await memory.count() >= 1

    async def test_manager_get_tools(self, memory):
        manager = MemoryManager(memory=memory)
        tools = manager.get_tools()
        names = [t.name for t in tools]
        assert "save_to_memory" in names
        assert "search_memory" in names
        assert "forget_memory" in names

    async def test_manager_custom_injection_strategy(self, memory, state):
        manager = MemoryManager(
            memory=memory,
            injection_strategy=UserMessageInjection(max_tokens=500),
        )
        await memory.add("Custom context")
        await manager.inject(state, "query")
        # UserMessageInjection appends to last user message
        last_user = next((m for m in reversed(state.messages) if m.role == "user"), None)
        assert last_user is not None
        assert "Custom context" in (last_user.text or "")

    async def test_manager_custom_save_strategy(self, memory, state):
        manager = MemoryManager(memory=memory, save_strategy=SaveEverythingStrategy())
        state.add_message(Message.assistant("Hi"))
        await manager.on_run_end("Hello", "Hi", state)
        await manager.on_tool_result("calculator", {"x": 1}, "42", state)
        assert await memory.count() >= 3

    async def test_manager_custom_query_strategy(self, memory, state):
        manager = MemoryManager(
            memory=memory,
            query_strategy=AdaptiveTokenQuery(base_tokens=1000, min_tokens=100),
        )
        await memory.add("relevant")
        await manager.inject(state, "relevant")
        assert len(state.messages) >= 2

    async def test_manager_startup_shutdown(self, memory):
        manager = MemoryManager(memory=memory)
        await manager.startup()
        await manager.health_check()
        await manager.shutdown()

    async def test_no_injection(self, memory, state):
        manager = MemoryManager(memory=memory, injection_strategy=NoInjection())
        await memory.add("secret")
        await manager.inject(state, "query")
        assert len(state.messages) == 2  # unchanged

    async def test_manager_add_search_clear(self, memory):
        manager = MemoryManager(memory=memory)
        eid = await manager.add("direct add", metadata={"tag": "test"})
        assert eid
        results = await manager.search("direct", limit=5)
        assert len(results) >= 1
        await manager.clear()
        assert await manager.count() == 0

    async def test_manager_repr(self, memory):
        manager = MemoryManager(memory=memory)
        r = repr(manager)
        assert "MemoryManager" in r
        assert "ConversationMemory" in r
