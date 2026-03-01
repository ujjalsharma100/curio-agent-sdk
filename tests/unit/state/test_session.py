"""
Unit tests for curio_agent_sdk.core.state.session — Session, SessionStore, SessionManager.

Covers: Session dataclass, touch, InMemorySessionStore create/get/list/add_message/
get_messages/delete, SessionManager create/get/list/delete/add_message/get_messages.
"""

from datetime import datetime

import pytest

from curio_agent_sdk.core.state.session import (
    Session,
    SessionStore,
    InMemorySessionStore,
    SessionManager,
)
from curio_agent_sdk.models.llm import Message


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSession:
    def test_session_creation(self):
        """Session dataclass."""
        session = Session(id="s1", agent_id="agent-1")
        assert session.id == "s1"
        assert session.agent_id == "agent-1"
        assert session.metadata == {}
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)

    def test_session_touch(self):
        """touch() updates updated_at."""
        session = Session(id="s1", agent_id="a1")
        old = session.updated_at
        session.touch()
        assert session.updated_at >= old


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInMemorySessionStore:
    @pytest.mark.asyncio
    async def test_session_store_create(self):
        """Create new session."""
        store = InMemorySessionStore()
        session = await store.create_session("agent-x")
        assert session.id
        assert session.agent_id == "agent-x"
        session2 = await store.create_session("agent-x", metadata={"k": "v"})
        assert session2.metadata == {"k": "v"}

    @pytest.mark.asyncio
    async def test_session_store_get(self):
        """Retrieve session."""
        store = InMemorySessionStore()
        created = await store.create_session("agent-g")
        got = await store.get_session(created.id)
        assert got is not None
        assert got.id == created.id
        assert got.agent_id == created.agent_id
        assert await store.get_session("nonexistent") is None

    @pytest.mark.asyncio
    async def test_session_store_list(self):
        """List sessions for agent (most recent first)."""
        store = InMemorySessionStore()
        s1 = await store.create_session("agent-l")
        s2 = await store.create_session("agent-l")
        await store.create_session("other-agent")
        sessions = await store.list_sessions("agent-l")
        assert len(sessions) == 2
        assert {s.id for s in sessions} == {s1.id, s2.id}

    @pytest.mark.asyncio
    async def test_session_store_add_message(self):
        """Add message to session."""
        store = InMemorySessionStore()
        session = await store.create_session("agent-m")
        await store.add_message(session.id, Message.user("Hello"))
        await store.add_message(session.id, Message.assistant("Hi there"))
        msgs = await store.get_messages(session.id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hello"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Hi there"

    @pytest.mark.asyncio
    async def test_session_store_get_messages(self):
        """Retrieve session messages."""
        store = InMemorySessionStore()
        session = await store.create_session("agent-msg")
        await store.add_message(session.id, Message.user("1"))
        await store.add_message(session.id, Message.assistant("2"))
        await store.add_message(session.id, Message.user("3"))
        msgs = await store.get_messages(session.id, limit=2)
        assert len(msgs) <= 2
        msgs_all = await store.get_messages(session.id, limit=50)
        assert len(msgs_all) == 3

    @pytest.mark.asyncio
    async def test_session_store_delete(self):
        """Delete session."""
        store = InMemorySessionStore()
        session = await store.create_session("agent-d")
        await store.add_message(session.id, Message.user("x"))
        deleted = await store.delete_session(session.id)
        assert deleted is True
        assert await store.get_session(session.id) is None
        assert await store.get_messages(session.id) == []
        assert await store.delete_session(session.id) is False


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionManager:
    @pytest.mark.asyncio
    async def test_session_manager_create(self):
        """Manager delegates create to store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        session = await manager.create("agent-mgr")
        assert session.agent_id == "agent-mgr"
        assert await store.get_session(session.id) is not None

    @pytest.mark.asyncio
    async def test_session_manager_get(self):
        """Manager delegates get to store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        created = await manager.create("agent-g")
        got = await manager.get(created.id)
        assert got is not None
        assert got.id == created.id
        assert await manager.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_session_manager_list(self):
        """Manager list delegates to store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        await manager.create("agent-list")
        await manager.create("agent-list")
        sessions = await manager.list("agent-list")
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_session_manager_delete(self):
        """Manager delete delegates to store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        session = await manager.create("agent-del")
        result = await manager.delete(session.id)
        assert result is True
        assert await manager.get(session.id) is None

    @pytest.mark.asyncio
    async def test_session_manager_add_message(self):
        """Manager add_message delegates to store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        session = await manager.create("agent-am")
        await manager.add_message(session.id, Message.user("from-manager"))
        msgs = await manager.get_messages(session.id)
        assert len(msgs) == 1
        assert msgs[0].content == "from-manager"

    @pytest.mark.asyncio
    async def test_session_manager_get_messages(self):
        """Manager get_messages delegates to store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        session = await manager.create("agent-gm")
        await manager.add_message(session.id, Message.assistant("reply"))
        msgs = await manager.get_messages(session.id)
        assert len(msgs) == 1
        assert msgs[0].content == "reply"

    def test_session_manager_store_property(self):
        """Manager exposes store via .store."""
        store = InMemorySessionStore()
        manager = SessionManager(store)
        assert manager.store is store
