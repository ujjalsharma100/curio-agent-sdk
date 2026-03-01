"""
Unit tests for curio_agent_sdk.memory.self_editing — SelfEditingMemory.
"""

import pytest

from curio_agent_sdk.memory.self_editing import SelfEditingMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory


@pytest.mark.unit
@pytest.mark.asyncio
class TestSelfEditingMemory:
    async def test_core_memory(self):
        mem = SelfEditingMemory()
        assert mem.core_memory_text == ""
        # Use add to archival; core is via tools
        await mem.add("archival entry", metadata={"type": "archival"})
        results = await mem.search("archival", limit=5)
        assert len(results) >= 1

    async def test_archival_memory(self):
        mem = SelfEditingMemory()
        eid = await mem.add("stored in archival", metadata={})
        assert eid
        found = await mem.get(eid)
        assert found is not None
        assert "stored in archival" in found.content

    async def test_memory_editing_via_tools(self):
        mem = SelfEditingMemory()
        tools = mem.get_tools()
        names = [t.name for t in tools]
        assert "core_memory_read" in names
        assert "core_memory_write" in names
        assert "core_memory_replace" in names
        assert "archival_memory_search" in names
        assert "archival_memory_insert" in names
        # Run core_memory_write
        write_tool = next(t for t in tools if t.name == "core_memory_write")
        result = await write_tool.execute(addition="User name is Bob")
        assert "updated" in result.lower() or "Core memory" in result
        assert "Bob" in mem.core_memory_text
