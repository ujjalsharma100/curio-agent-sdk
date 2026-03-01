"""
Performance tests: Memory Operations (Phase 19)

Validates memory add/search throughput at scale.
"""

import asyncio
import time
import pytest

from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_add_throughput():
    """1000 memory additions complete in < 5s."""
    memory = ConversationMemory(max_entries=2000)

    start = time.monotonic()
    for i in range(1000):
        await memory.add(f"Entry number {i} with some content about topic {i % 10}")
    elapsed = time.monotonic() - start

    count = await memory.count()
    assert count == 1000
    assert elapsed < 5.0, f"1000 memory adds took {elapsed:.2f}s (limit: 5s)"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_search_throughput():
    """1000 memory searches complete in < 5s."""
    memory = ConversationMemory(max_entries=500)

    # Pre-populate with 500 entries
    for i in range(500):
        await memory.add(
            f"Topic {i % 20}: detail about item {i}",
            metadata={"category": f"cat_{i % 10}"},
        )

    start = time.monotonic()
    for i in range(1000):
        results = await memory.search(f"Topic {i % 20}", limit=5)
        assert isinstance(results, list)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"1000 memory searches took {elapsed:.2f}s (limit: 5s)"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_kv_memory_throughput():
    """1000 KV set/get operations complete in < 5s."""
    kv = KeyValueMemory()

    start = time.monotonic()
    for i in range(1000):
        await kv.set(f"key_{i}", f"value_{i}")

    for i in range(1000):
        val = await kv.get_value(f"key_{i}")
        assert val == f"value_{i}"
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"2000 KV ops took {elapsed:.2f}s (limit: 5s)"
