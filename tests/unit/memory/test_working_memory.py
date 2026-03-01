"""
Unit tests for curio_agent_sdk.memory.working — WorkingMemory.
"""

import pytest

from curio_agent_sdk.memory.working import WorkingMemory


@pytest.mark.unit
@pytest.mark.asyncio
class TestWorkingMemory:
    async def test_ephemeral_storage(self):
        mem = WorkingMemory()
        await mem.write("goal", "Implement auth")
        val = await mem.read("goal")
        assert val == "Implement auth"

    async def test_scratchpad_operations(self):
        mem = WorkingMemory()
        await mem.write("note1", "first note")
        await mem.write("note2", "second note")
        assert await mem.read("note1") == "first note"
        assert await mem.read("note2") == "second note"
        await mem.delete("note1")
        assert await mem.read("note1") is None
        assert await mem.count() == 1

    async def test_add_with_key_metadata(self):
        mem = WorkingMemory()
        await mem.add("content", metadata={"key": "scratch"})
        assert await mem.read("scratch") == "content"
