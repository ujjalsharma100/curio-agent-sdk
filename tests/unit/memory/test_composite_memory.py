"""
Unit tests for curio_agent_sdk.memory.composite — CompositeMemory.
"""

import pytest

from curio_agent_sdk.memory.composite import CompositeMemory
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.base import MemoryEntry


@pytest.mark.unit
@pytest.mark.asyncio
class TestCompositeMemory:
    async def test_composite_add(self):
        mem = CompositeMemory({
            "conv": ConversationMemory(max_entries=10),
            "kv": KeyValueMemory(),
        })
        eid = await mem.add("hello")
        assert eid
        assert await mem.count() >= 2  # added to both

    async def test_composite_search(self):
        mem = CompositeMemory({
            "conv": ConversationMemory(max_entries=10),
            "kv": KeyValueMemory(),
        })
        await mem.add("searchable content")
        results = await mem.search("searchable", limit=5)
        assert isinstance(results, list)
        assert len(results) >= 1

    async def test_composite_priority(self):
        """Search merges and sorts by relevance across sub-memories."""
        mem = CompositeMemory({
            "a": ConversationMemory(max_entries=5),
            "b": KeyValueMemory(),
        })
        await mem.add("term in both")
        results = await mem.search("term", limit=10)
        # Deduplicated by content
        contents = [r.content for r in results]
        assert "term in both" in contents

    async def test_composite_get_memory(self):
        conv = ConversationMemory(max_entries=5)
        mem = CompositeMemory({"conv": conv, "kv": KeyValueMemory()})
        assert mem.get_memory("conv") is conv

    async def test_composite_add_with_targets(self):
        conv = ConversationMemory(max_entries=5)
        kv = KeyValueMemory()
        mem = CompositeMemory({"conv": conv, "kv": kv})
        await mem.add("only in conv", metadata={"memory_targets": ["conv"]})
        assert await conv.count() == 1
        assert await kv.count() == 0
