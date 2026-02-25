"""
Base memory interface and data models.

Memory provides agents with the ability to store and retrieve information
across conversation turns and sessions.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    """
    A single entry in agent memory.

    Each entry has content, optional metadata, a relevance score
    (populated during search), and timestamps for lifecycle tracking.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance: float = 0.0  # Populated during search
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "relevance": self.relevance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        entry = cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            relevance=data.get("relevance", 0.0),
        )
        if data.get("created_at"):
            entry.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            entry.updated_at = datetime.fromisoformat(data["updated_at"])
        return entry


class Memory(ABC):
    """
    Abstract base class for agent memory.

    Memory allows agents to store and retrieve information for use
    in future reasoning steps. Different implementations provide
    different storage and retrieval strategies:

    - ConversationMemory: Recent conversation turns (sliding window)
    - VectorMemory: Semantic search using embeddings
    - KeyValueMemory: Structured key-value store
    - CompositeMemory: Combines multiple memory types

    Example:
        class MyAgent:
            def __init__(self):
                self.memory = ConversationMemory(max_entries=100)

            async def process(self, input):
                # Retrieve relevant context
                context = await self.memory.get_context(input, max_tokens=2000)

                # ... use context in LLM call ...

                # Store new information
                await self.memory.add(f"User asked: {input}")
                await self.memory.add(f"I answered: {response}")
    """

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Store a memory entry.

        Args:
            content: The text content to store.
            metadata: Optional metadata (tags, source, type, etc.)

        Returns:
            The ID of the stored entry.
        """
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Search memories by relevance to a query.

        Args:
            query: The search query.
            limit: Maximum number of results to return.

        Returns:
            List of MemoryEntry objects sorted by relevance (highest first).
        """
        ...

    @abstractmethod
    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get relevant context formatted for prompt inclusion.

        This is the primary method used by agents to augment their
        context with memory. Returns a formatted string that can be
        injected into the system prompt or conversation.

        Args:
            query: The current query/objective for relevance.
            max_tokens: Approximate max tokens for the returned context.

        Returns:
            Formatted string of relevant memories.
        """
        ...

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """
        Get a specific memory entry by ID.

        Default implementation searches all entries. Override for
        more efficient implementations.
        """
        return None

    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry by ID.

        Returns True if the entry was found and deleted.
        """
        return False

    async def clear(self) -> None:
        """Clear all memory entries."""
        pass

    async def count(self) -> int:
        """Return the number of stored entries."""
        return 0
