"""
Unit tests for curio_agent_sdk.memory.file_memory — FileMemory.
"""

import pytest

from curio_agent_sdk.memory.file_memory import FileMemory


@pytest.mark.unit
@pytest.mark.asyncio
class TestFileMemory:
    async def test_file_persistence(self, tmp_path):
        mem = FileMemory(memory_dir=tmp_path / "mem")
        await mem.startup()
        try:
            eid = await mem.add("User prefers dark mode")
            assert eid
            ctx = await mem.get_context("preferences", max_tokens=500)
            assert "dark mode" in ctx or "prefers" in ctx
            # Restart: new instance, load from disk
            mem2 = FileMemory(memory_dir=tmp_path / "mem")
            await mem2.startup()
            try:
                count = await mem2.count()
                assert count >= 1
                entry = await mem2.get(eid)
                assert entry is not None
                assert "dark mode" in entry.content
            finally:
                await mem2.shutdown()
        finally:
            await mem.shutdown()

    async def test_file_not_found(self, tmp_path):
        mem = FileMemory(memory_dir=tmp_path / "mem")
        await mem.startup()
        try:
            entry = await mem.get("nonexistent_id_12345")
            assert entry is None
            deleted = await mem.delete("nonexistent_id_12345")
            assert deleted is False
        finally:
            await mem.shutdown()

    async def test_concurrent_access(self, tmp_path):
        """Sequential add/read from same instance (no real concurrency)."""
        mem = FileMemory(memory_dir=tmp_path / "mem")
        await mem.startup()
        try:
            ids = []
            for i in range(5):
                eid = await mem.add(f"entry {i}")
                ids.append(eid)
            for i, eid in enumerate(ids):
                entry = await mem.get(eid)
                assert entry is not None
                assert f"entry {i}" in entry.content
        finally:
            await mem.shutdown()
