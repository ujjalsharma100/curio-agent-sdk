"""
Unit tests for curio_agent_sdk.core.state.state_store — StateStore implementations.

Covers: InMemoryStateStore save/load/list/delete, FileStateStore save/load/list/delete,
corrupted file handling.
"""

import json
import pytest

from curio_agent_sdk.core.state.state import AgentState
from curio_agent_sdk.core.state.state_store import InMemoryStateStore, FileStateStore
from curio_agent_sdk.models.llm import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(messages=None, iteration=0, metadata=None):
    """Create an AgentState with optional messages and metadata."""
    state = AgentState(
        messages=list(messages or []),
        iteration=iteration,
        metadata=dict(metadata or {}),
    )
    return state


# ---------------------------------------------------------------------------
# InMemoryStateStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInMemoryStateStore:
    @pytest.mark.asyncio
    async def test_inmemory_store_save(self):
        """Save state."""
        store = InMemoryStateStore()
        state = _make_state(
            messages=[Message.user("hi"), Message.assistant("hello")],
            iteration=2,
            metadata={"k": "v"},
        )
        await store.save(state, run_id="run-1", agent_id="agent-a")
        loaded = await store.load("run-1")
        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "hi"
        assert loaded.messages[1].content == "hello"
        assert loaded.iteration == 2
        assert loaded.metadata == {"k": "v"}

    @pytest.mark.asyncio
    async def test_inmemory_store_load(self):
        """Load saved state."""
        store = InMemoryStateStore()
        state = _make_state(messages=[Message.system("sys")], iteration=1)
        await store.save(state, run_id="r1", agent_id="a1")
        loaded = await store.load("r1")
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "sys"
        assert loaded.iteration == 1

    @pytest.mark.asyncio
    async def test_inmemory_store_load_nonexistent(self):
        """Returns None for missing run."""
        store = InMemoryStateStore()
        assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_inmemory_store_list_runs(self):
        """List runs for agent (most recent first)."""
        store = InMemoryStateStore()
        await store.save(_make_state(iteration=1), "run-1", "agent-x")
        await store.save(_make_state(iteration=2), "run-2", "agent-x")
        await store.save(_make_state(iteration=3), "run-3", "agent-y")
        runs_x = await store.list_runs("agent-x")
        runs_y = await store.list_runs("agent-y")
        assert set(runs_x) == {"run-1", "run-2"}
        assert runs_y == ["run-3"]

    @pytest.mark.asyncio
    async def test_inmemory_store_delete(self):
        """Delete a run."""
        store = InMemoryStateStore()
        await store.save(_make_state(), "run-1", "agent-a")
        assert await store.load("run-1") is not None
        deleted = await store.delete("run-1")
        assert deleted is True
        assert await store.load("run-1") is None
        assert await store.delete("run-1") is False


# ---------------------------------------------------------------------------
# FileStateStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFileStateStore:
    @pytest.mark.asyncio
    async def test_file_store_save_load(self, tmp_path):
        """File-based save/load roundtrip."""
        store = FileStateStore(tmp_path)
        state = _make_state(
            messages=[Message.user("file-test"), Message.assistant("ok")],
            iteration=3,
            metadata={"source": "file"},
        )
        await store.save(state, run_id="file-run-1", agent_id="agent-f")
        loaded = await store.load("file-run-1")
        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "file-test"
        assert loaded.messages[1].content == "ok"
        assert loaded.iteration == 3
        assert loaded.metadata == {"source": "file"}

    @pytest.mark.asyncio
    async def test_file_store_list_runs(self, tmp_path):
        """List runs from files."""
        store = FileStateStore(tmp_path)
        await store.save(_make_state(iteration=1), "f1", "ag")
        await store.save(_make_state(iteration=2), "f2", "ag")
        await store.save(_make_state(iteration=3), "f3", "other")
        runs = await store.list_runs("ag")
        assert set(runs) == {"f1", "f2"}
        assert await store.list_runs("other") == ["f3"]

    @pytest.mark.asyncio
    async def test_file_store_delete(self, tmp_path):
        """Delete file-based run."""
        store = FileStateStore(tmp_path)
        await store.save(_make_state(), "del-run", "agent-d")
        assert (tmp_path / "del-run.json").exists()
        deleted = await store.delete("del-run")
        assert deleted is True
        assert not (tmp_path / "del-run.json").exists()
        assert await store.load("del-run") is None
        assert await store.delete("del-run") is False

    @pytest.mark.asyncio
    async def test_file_store_corrupted_file(self, tmp_path):
        """Handle corrupted state file (load returns None)."""
        store = FileStateStore(tmp_path)
        bad_path = tmp_path / "bad-run.json"
        bad_path.write_text("not valid json {{{")
        loaded = await store.load("bad-run")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_file_store_load_nonexistent(self, tmp_path):
        """Load returns None when file does not exist."""
        store = FileStateStore(tmp_path)
        assert await store.load("no-such-run") is None
