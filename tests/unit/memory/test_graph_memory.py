"""
Unit tests for curio_agent_sdk.memory.graph — GraphMemory, Triple.
"""

import pytest

from curio_agent_sdk.memory.graph import GraphMemory, Triple


@pytest.mark.unit
@pytest.mark.asyncio
class TestGraphMemory:
    async def test_add_entity(self):
        mem = GraphMemory()
        await mem.add_entity("Alice", {"type": "person"})
        await mem.add_entity("Acme", {"type": "company"})
        await mem.add_relation("Alice", "works_at", "Acme")
        triples = await mem.query("Alice", limit=10)
        assert len(triples) >= 1
        assert any(t.subject == "Alice" and t.relation == "works_at" and t.obj == "Acme" for t in triples)

    async def test_add_relationship(self):
        mem = GraphMemory()
        await mem.add_relation("A", "knows", "B")
        await mem.add_relation("B", "knows", "C")
        q = await mem.query("knows", limit=10)
        assert len(q) == 2

    async def test_query_relationships(self):
        mem = GraphMemory()
        await mem.add_relation("X", "related_to", "Y")
        triples = await mem.query("X", limit=5)
        assert len(triples) >= 1
        assert triples[0].subject == "X"
        assert triples[0].obj == "Y"

    async def test_entity_context(self):
        mem = GraphMemory()
        await mem.add_relation("Alice", "works_at", "Acme")
        ctx = await mem.get_context("Alice", max_tokens=500)
        assert "Alice" in ctx
        assert "Acme" in ctx

    async def test_triple_to_dict_from_dict(self):
        t = Triple(subject="a", relation="r", obj="b", metadata={"k": "v"})
        d = t.to_dict()
        assert d["subject"] == "a"
        assert d["relation"] == "r"
        assert d["object"] == "b"
        t2 = Triple.from_dict(d)
        assert t2.subject == t.subject and t2.obj == t.obj
