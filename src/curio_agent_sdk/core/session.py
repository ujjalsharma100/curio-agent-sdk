"""
Session and conversation management.

Provides persistent conversation sessions with history management,
session persistence, and optional conversation branching.

Example:
    store = InMemorySessionStore()
    session_mgr = SessionManager(store)
    session = await session_mgr.create(agent.agent_id)
    result = await agent.arun("Hello", session_id=session.id)
    result = await agent.arun("Follow up", session_id=session.id)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from curio_agent_sdk.models.llm import Message

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    A persistent conversation session.

    Holds conversation history and metadata for multi-turn interactions
    with an agent. Messages are stored separately via SessionStore;
    this model is the session metadata and identity.
    """
    id: str
    agent_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Serialize a Message to a JSON-compatible dict (matches checkpoint format)."""
    from curio_agent_sdk.core.checkpoint import _serialize_message as _cp_serialize
    return _cp_serialize(msg)


def _deserialize_message(data: dict[str, Any]) -> Message:
    """Deserialize a Message from a dict (matches checkpoint format)."""
    from curio_agent_sdk.core.checkpoint import _deserialize_message as _cp_deserialize
    return _cp_deserialize(data)


class SessionStore(ABC):
    """
    Abstract backend for persisting sessions and their messages.

    Implementations can store in memory, files, or databases.
    """

    @abstractmethod
    async def create_session(self, agent_id: str, metadata: dict[str, Any] | None = None) -> Session:
        """Create a new session for the given agent. Returns the created Session."""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None:
        """Load a session by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def list_sessions(self, agent_id: str, limit: int = 50) -> list[Session]:
        """List sessions for an agent (most recent first)."""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages. Returns True if deleted."""
        ...

    @abstractmethod
    async def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to the session's conversation history."""
        ...

    @abstractmethod
    async def get_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        """Get the most recent messages for a session (oldest first, up to limit)."""
        ...


class InMemorySessionStore(SessionStore):
    """
    In-memory session store for testing and development.

    Sessions and messages are lost when the process exits.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}  # session_id -> list of serialized messages

    async def create_session(self, agent_id: str, metadata: dict[str, Any] | None = None) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            agent_id=agent_id,
            metadata=dict(metadata or {}),
        )
        self._sessions[session_id] = session
        self._messages[session_id] = []
        return session

    async def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def list_sessions(self, agent_id: str, limit: int = 50) -> list[Session]:
        sessions = [s for s in self._sessions.values() if s.agent_id == agent_id]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._messages.pop(session_id, None)
            return True
        return False

    async def add_message(self, session_id: str, message: Message) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(_serialize_message(message))
        session = self._sessions.get(session_id)
        if session:
            session.touch()

    async def get_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        raw = self._messages.get(session_id, [])
        # Return oldest-first, up to limit (tail of list)
        to_load = raw[-limit:] if limit else raw
        return [_deserialize_message(m) for m in to_load]


class SessionManager:
    """
    Manages conversation sessions with a pluggable store.

    Use this when you want multi-turn conversations where history
    is persisted across agent.arun() calls. Pass session_manager to
    the agent (or builder) and session_id to arun().
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    @property
    def store(self) -> SessionStore:
        """The underlying session store."""
        return self._store

    async def create(self, agent_id: str, metadata: dict[str, Any] | None = None) -> Session:
        """Create a new session for the given agent."""
        return await self._store.create_session(agent_id, metadata=metadata)

    async def get(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return await self._store.get_session(session_id)

    async def list(self, agent_id: str, limit: int = 50) -> list[Session]:
        """List sessions for an agent (most recent first)."""
        return await self._store.list_sessions(agent_id, limit=limit)

    async def delete(self, session_id: str) -> bool:
        """Delete a session and its message history."""
        return await self._store.delete_session(session_id)

    async def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to a session's conversation history."""
        await self._store.add_message(session_id, message)

    async def get_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        """Get the most recent messages for a session (oldest first)."""
        return await self._store.get_messages(session_id, limit=limit)
