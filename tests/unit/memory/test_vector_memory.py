"""
Unit tests for curio_agent_sdk.memory.vector — VectorMemory.
"""

import pytest

from curio_agent_sdk.memory.vector import VectorMemory


async def _mock_embedding(texts: list[str]) -> list[list[float]]:
    """Mock embedding: same text -> same vector (deterministic)."""
    out = []
    for t in texts:
        h = hash(t) % 10000
        out.append([float((h + i) % 100) / 100.0 for i in range(8)])
    return out


@pytest.mark.unit
@pytest.mark.asyncio
class TestVectorMemory:
    async def test_embedding_storage(self):
        mem = VectorMemory(embedding_fn=_mock_embedding)
        await mem.startup()
        try:
            eid = await mem.add("Quantum computing uses qubits")
            assert eid
            assert await mem.count() == 1
        finally:
            await mem.shutdown()

    async def test_semantic_search(self):
        mem = VectorMemory(embedding_fn=_mock_embedding)
        await mem.startup()
        try:
            await mem.add("Quantum computing uses qubits")
            await mem.add("Weather is sunny today")
            results = await mem.search("What are qubits?", limit=5)
            assert len(results) >= 1
            # Same mock gives deterministic similarity
            assert any("qubit" in e.content.lower() for e in results)
        finally:
            await mem.shutdown()

    async def test_mock_embeddings(self):
        """Works without real embeddings (mock only)."""
        mem = VectorMemory(embedding_fn=_mock_embedding)
        await mem.add("first")
        await mem.add("second")
        ctx = await mem.get_context("first", max_tokens=200)
        assert isinstance(ctx, str)

    async def test_add_batch(self):
        mem = VectorMemory(embedding_fn=_mock_embedding)
        ids = await mem.add_batch([
            ("content one", None),
            ("content two", {"tag": "b"}),
        ])
        assert len(ids) == 2
        assert await mem.count() == 2
