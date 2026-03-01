"""
Unit tests for curio_agent_sdk.memory.conversation — ConversationMemory.
"""

import pytest

from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.base import MemoryEntry


@pytest.mark.unit
@pytest.mark.asyncio
class TestConversationMemory:
    async def test_sliding_window(self):
        mem = ConversationMemory(max_entries=3)
        id1 = await mem.add("first")
        id2 = await mem.add("second")
        id3 = await mem.add("third")
        assert await mem.count() == 3
        # Add one more; oldest should be evicted
        id4 = await mem.add("fourth")
        assert await mem.count() == 3
        assert await mem.get(id1) is None
        assert await mem.get(id2) is not None
        assert await mem.get(id4) is not None

    async def test_message_ordering(self):
        mem = ConversationMemory(max_entries=10)
        await mem.add("oldest")
        await mem.add("middle")
        await mem.add("newest")
        results = await mem.search("", limit=10)
        # Most recent should rank higher (recency score)
        contents = [r.content for r in results]
        assert "newest" in contents
        assert "oldest" in contents

    async def test_window_overflow(self):
        mem = ConversationMemory(max_entries=2)
        await mem.add("a")
        await mem.add("b")
        await mem.add("c")
        assert await mem.count() == 2
        entries = await mem.search("", limit=5)
        assert len(entries) == 2
        contents = {e.content for e in entries}
        assert "c" in contents
        assert "b" in contents
        assert "a" not in contents

    async def test_get_recent(self):
        mem = ConversationMemory(max_entries=10)
        await mem.add("1")
        await mem.add("2")
        await mem.add("3")
        recent = mem.get_recent(n=2)
        assert len(recent) == 2
        assert recent[0].content == "3"
        assert recent[1].content == "2"
