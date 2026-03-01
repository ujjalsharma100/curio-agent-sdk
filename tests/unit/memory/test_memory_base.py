"""
Unit tests for curio_agent_sdk.memory.base — Memory ABC contract.

Parametrized over all Memory implementations to ensure consistent behavior.
"""

import pytest

from pathlib import Path

import pytest

from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.working import WorkingMemory
from curio_agent_sdk.memory.episodic import EpisodicMemory
from curio_agent_sdk.memory.graph import GraphMemory
from curio_agent_sdk.memory.self_editing import SelfEditingMemory
from curio_agent_sdk.memory.composite import CompositeMemory
def _make_memory_impls(tmp_path: Path):
    """Build in-memory implementations (no Component startup needed for contract tests)."""
    return [
        ("ConversationMemory", ConversationMemory(max_entries=20)),
        ("KeyValueMemory", KeyValueMemory()),
        ("WorkingMemory", WorkingMemory()),
        ("EpisodicMemory", EpisodicMemory(max_episodes=20)),
        ("GraphMemory", GraphMemory()),
        ("SelfEditingMemory", SelfEditingMemory()),
    ]


@pytest.fixture(params=list(range(6)), ids=[
    "ConversationMemory", "KeyValueMemory", "WorkingMemory", "EpisodicMemory",
    "GraphMemory", "SelfEditingMemory",
])
def memory_impl(request, tmp_path):
    """Provide each in-memory Memory implementation (File/Vector tested in their own modules)."""
    impls = _make_memory_impls(tmp_path)
    _name, mem = impls[request.param]
    return mem


@pytest.fixture
def memory(memory_impl):
    """Alias for memory_impl for tests that need a fresh instance."""
    return memory_impl


# ─── MemoryEntry tests ────────────────────────────────────────────────────


@pytest.mark.unit
class TestMemoryEntry:
    def test_memory_entry_creation(self):
        entry = MemoryEntry(content="hello", metadata={"k": "v"})
        assert entry.content == "hello"
        assert entry.metadata == {"k": "v"}
        assert entry.id
        assert len(entry.id) == 12
        assert entry.relevance == 0.0
        assert entry.created_at is not None
        assert entry.updated_at is not None

    def test_memory_entry_to_dict(self):
        entry = MemoryEntry(id="abc", content="x", metadata={})
        d = entry.to_dict()
        assert d["id"] == "abc"
        assert d["content"] == "x"
        assert "created_at" in d
        assert "updated_at" in d

    def test_memory_entry_from_dict(self):
        d = {
            "id": "xyz",
            "content": "test",
            "metadata": {"a": 1},
            "relevance": 0.8,
            "created_at": "2025-01-01T12:00:00",
            "updated_at": "2025-01-01T12:00:00",
        }
        entry = MemoryEntry.from_dict(d)
        assert entry.id == "xyz"
        assert entry.content == "test"
        assert entry.metadata == {"a": 1}
        assert entry.relevance == 0.8


# ─── Contract tests (parametrized over memory implementations) ────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryContract:
    async def test_memory_add(self, memory):
        entry_id = await memory.add("first entry", metadata={"tag": "a"})
        assert entry_id
        assert isinstance(entry_id, str)

    async def test_memory_search(self, memory):
        await memory.add("hello world", metadata={})
        await memory.add("foo bar", metadata={})
        results = await memory.search("hello", limit=5)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, MemoryEntry)
            assert hasattr(r, "id")
            assert hasattr(r, "content")
            assert hasattr(r, "relevance")

    async def test_memory_get_context(self, memory):
        await memory.add("some context here", metadata={})
        ctx = await memory.get_context("context", max_tokens=500)
        assert isinstance(ctx, str)

    async def test_memory_get(self, memory):
        eid = await memory.add("get me", metadata={})
        entry = await memory.get(eid)
        if entry is not None:
            assert entry.id == eid
            # Content may be stored as-is or as triple (e.g. GraphMemory: "get related_to me")
            assert "get" in entry.content and "me" in entry.content

    async def test_memory_delete(self, memory):
        eid = await memory.add("to delete", metadata={})
        deleted = await memory.delete(eid)
        assert isinstance(deleted, bool)
        if deleted:
            entry = await memory.get(eid)
            assert entry is None

    async def test_memory_clear(self, memory):
        await memory.add("a", metadata={})
        await memory.add("b", metadata={})
        await memory.clear()
        count = await memory.count()
        assert count == 0

    async def test_memory_count(self, memory):
        await memory.clear()
        assert await memory.count() == 0
        await memory.add("one", metadata={})
        assert await memory.count() >= 1

    async def test_memory_empty_search(self, memory):
        await memory.clear()
        results = await memory.search("anything", limit=5)
        assert results == []


# ─── CompositeMemory needs sub-memories; test with simple in-memory ────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryContractComposite:
    @pytest.fixture
    def memory(self):
        return CompositeMemory({
            "conv": ConversationMemory(max_entries=10),
            "kv": KeyValueMemory(),
        })

    async def test_composite_add(self, memory):
        eid = await memory.add("composite entry", metadata={})
        assert eid

    async def test_composite_search(self, memory):
        await memory.add("searchable", metadata={})
        results = await memory.search("searchable", limit=5)
        assert isinstance(results, list)

    async def test_composite_get_context(self, memory):
        await memory.add("context", metadata={})
        ctx = await memory.get_context("context", max_tokens=500)
        assert isinstance(ctx, str)

    async def test_composite_clear(self, memory):
        await memory.add("x", metadata={})
        await memory.clear()
        assert await memory.count() == 0
