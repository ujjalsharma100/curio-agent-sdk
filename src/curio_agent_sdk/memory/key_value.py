"""
Key-value memory - simple structured data store.

Stores and retrieves memories by key. Useful for storing
structured facts, preferences, and named data that the agent
needs to recall by name rather than by semantic similarity.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from curio_agent_sdk.memory.base import Memory, MemoryEntry

logger = logging.getLogger(__name__)

# Approximate chars per token
_CHARS_PER_TOKEN = 4


class KeyValueMemory(Memory):
    """
    Simple key-value memory store.

    Stores entries with explicit keys for direct retrieval. Search
    matches against both keys and content using simple string matching.

    Useful for:
    - Storing user preferences ("user_name" -> "Alice")
    - Caching computed facts ("weather_sf" -> "72°F, sunny")
    - Maintaining structured state across turns

    Example:
        memory = KeyValueMemory()

        await memory.set("user_name", "Alice")
        await memory.set("preference_language", "Python")

        name = await memory.get_value("user_name")  # "Alice"
        context = await memory.get_context("user preferences")
    """

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Add an entry. If metadata contains a "key" field, it's used
        as the storage key; otherwise the entry ID is used.
        """
        meta = metadata or {}
        key = meta.get("key")

        entry = MemoryEntry(content=content, metadata=meta)

        if key:
            entry.id = key
            # Update existing if key exists
            if key in self._store:
                entry.created_at = self._store[key].created_at
            self._store[key] = entry
        else:
            self._store[entry.id] = entry

        return entry.id

    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Set a key-value pair (convenience method).

        Args:
            key: The storage key.
            value: The value to store.
            metadata: Optional additional metadata.

        Returns:
            The key.
        """
        meta = {**(metadata or {}), "key": key}
        return await self.add(value, metadata=meta)

    async def get_value(self, key: str) -> str | None:
        """
        Get a value by key (convenience method).

        Returns the content string or None if not found.
        """
        entry = self._store.get(key)
        return entry.content if entry else None

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Search by matching query against keys and content.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        results: list[MemoryEntry] = []
        for key, entry in self._store.items():
            # Score based on key and content matches
            key_lower = key.lower()
            content_lower = entry.content.lower()
            combined = f"{key_lower} {content_lower}"

            hits = sum(1 for term in query_terms if term in combined)
            if hits == 0:
                continue

            relevance = hits / max(len(query_terms), 1)

            # Boost exact key matches
            if query_lower in key_lower or key_lower in query_lower:
                relevance = min(1.0, relevance + 0.5)

            result = MemoryEntry(
                id=entry.id,
                content=entry.content,
                metadata=entry.metadata,
                relevance=relevance,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
            )
            results.append(result)

        results.sort(key=lambda e: e.relevance, reverse=True)
        return results[:limit]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get key-value pairs formatted for prompt inclusion.
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN

        results = await self.search(query, limit=20)

        if not results:
            # If no search hits, return all entries that fit
            results = [
                MemoryEntry(
                    id=e.id, content=e.content, metadata=e.metadata,
                    created_at=e.created_at, updated_at=e.updated_at,
                )
                for e in self._store.values()
            ]

        selected: list[MemoryEntry] = []
        total_chars = 0

        for entry in results:
            key = entry.metadata.get("key", entry.id)
            line = f"{key}: {entry.content}"
            entry_chars = len(line) + 5
            if total_chars + entry_chars > max_chars:
                break
            selected.append(entry)
            total_chars += entry_chars

        if not selected:
            return ""

        lines = ["[Stored Facts]"]
        for entry in selected:
            key = entry.metadata.get("key", entry.id)
            lines.append(f"- {key}: {entry.content}")
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        return self._store.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        return self._store.pop(entry_id, None) is not None

    async def clear(self) -> None:
        self._store.clear()

    async def count(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        """Get all stored keys (sync, for convenience)."""
        return list(self._store.keys())
