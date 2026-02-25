"""
Conversation memory - short-term memory with sliding window.

Stores recent conversation turns and retrieves them by recency.
Useful for maintaining context in multi-turn conversations.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from curio_agent_sdk.memory.base import Memory, MemoryEntry

logger = logging.getLogger(__name__)

# Approximate chars per token for budget estimation
_CHARS_PER_TOKEN = 4


class ConversationMemory(Memory):
    """
    Short-term conversation memory with a sliding window.

    Stores the most recent entries up to `max_entries`. Older entries
    are automatically evicted when the limit is reached.

    Search returns entries by recency (most recent first), with
    optional keyword filtering.

    Example:
        memory = ConversationMemory(max_entries=50)

        await memory.add("User asked about quantum computing")
        await memory.add("I explained the basics of qubits")

        # Get recent context
        context = await memory.get_context("quantum", max_tokens=1000)
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._index: dict[str, MemoryEntry] = {}

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
        )
        # If deque is full, the oldest entry will be auto-evicted
        if len(self._entries) == self.max_entries:
            evicted = self._entries[0]
            self._index.pop(evicted.id, None)

        self._entries.append(entry)
        self._index[entry.id] = entry
        return entry.id

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Search by recency with keyword relevance boosting.

        Entries containing query terms get higher relevance scores.
        Results are sorted by relevance (recency + keyword match).
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        results: list[MemoryEntry] = []
        entries = list(self._entries)

        for i, entry in enumerate(reversed(entries)):
            # Base recency score: most recent = highest
            recency_score = 1.0 - (i / max(len(entries), 1))

            # Keyword relevance boost
            content_lower = entry.content.lower()
            keyword_hits = sum(1 for term in query_terms if term in content_lower)
            keyword_score = keyword_hits / max(len(query_terms), 1)

            # Combined score
            relevance = 0.5 * recency_score + 0.5 * keyword_score

            result = MemoryEntry(
                id=entry.id,
                content=entry.content,
                metadata=entry.metadata,
                relevance=relevance,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
            )
            results.append(result)

        # Sort by relevance (highest first)
        results.sort(key=lambda e: e.relevance, reverse=True)
        return results[:limit]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get recent conversation context formatted for prompt inclusion.

        Returns entries in chronological order (oldest first) that fit
        within the token budget.
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN
        entries = list(self._entries)

        # Build context from most recent, respecting token budget
        selected: list[MemoryEntry] = []
        total_chars = 0

        for entry in reversed(entries):
            entry_chars = len(entry.content) + 20  # overhead for formatting
            if total_chars + entry_chars > max_chars:
                break
            selected.append(entry)
            total_chars += entry_chars

        if not selected:
            return ""

        # Reverse back to chronological order
        selected.reverse()

        lines = ["[Conversation Memory]"]
        for entry in selected:
            lines.append(f"- {entry.content}")
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        return self._index.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        entry = self._index.pop(entry_id, None)
        if entry is not None:
            self._entries.remove(entry)
            return True
        return False

    async def clear(self) -> None:
        self._entries.clear()
        self._index.clear()

    async def count(self) -> int:
        return len(self._entries)

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get the N most recent entries (sync, for convenience)."""
        entries = list(self._entries)
        return list(reversed(entries[-n:]))
