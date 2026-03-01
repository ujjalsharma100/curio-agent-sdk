"""
Unit tests for curio_agent_sdk.memory.key_value — KeyValueMemory.
"""

import pytest

from curio_agent_sdk.memory.key_value import KeyValueMemory


@pytest.mark.unit
@pytest.mark.asyncio
class TestKeyValueMemory:
    async def test_kv_set_get(self):
        mem = KeyValueMemory()
        await mem.set("user_name", "Alice")
        val = await mem.get_value("user_name")
        assert val == "Alice"

    async def test_kv_update(self):
        mem = KeyValueMemory()
        await mem.set("key", "value1")
        await mem.set("key", "value2")
        assert await mem.get_value("key") == "value2"
        assert await mem.count() == 1

    async def test_kv_delete_key(self):
        mem = KeyValueMemory()
        await mem.set("k", "v")
        deleted = await mem.delete("k")
        assert deleted is True
        assert await mem.get_value("k") is None
        assert await mem.delete("nonexistent") is False

    async def test_keys(self):
        mem = KeyValueMemory()
        await mem.set("a", "1")
        await mem.set("b", "2")
        k = mem.keys()
        assert set(k) == {"a", "b"}
