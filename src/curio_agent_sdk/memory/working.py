"""
Working memory — ephemeral scratchpad for the current task.

In-context, short-lived storage for the agent's current reasoning.
Cleared between runs unless explicitly persisted elsewhere.
"""

from __future__ import annotations

from typing import Any

from curio_agent_sdk.memory.base import Memory, MemoryEntry

# Approximate chars per token
_CHARS_PER_TOKEN = 4


class WorkingMemory(Memory):
    """
    Ephemeral, in-context scratchpad for the current task.

    Stores key-value pairs that are always returned as context
    (no semantic search). Useful for intermediate reasoning,
    current-session facts, and task-specific notes.

    Example:
        memory = WorkingMemory()
        await memory.write("current_goal", "Implement auth")
        await memory.read("current_goal")  # "Implement auth"
        context = await memory.get_context("anything", max_tokens=500)
    """

    def __init__(self):
        self._store: dict[str, str] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    async def write(self, key: str, value: str) -> None:
        """Write a key-value pair to working memory."""
        self._store[key] = value
        self._meta[key] = self._meta.get(key, {})

    async def read(self, key: str) -> str | None:
        """Read a value by key. Returns None if not found."""
        return self._store.get(key)

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Store content. If metadata has "key", use it for direct read/write;
        otherwise generate an id and store under that id.
        """
        meta = metadata or {}
        key = meta.get("key")
        if key is None:
            import uuid
            key = uuid.uuid4().hex[:12]
        self._store[key] = content
        self._meta[key] = {**meta, "key": key}
        return key

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Search by matching query against keys and values.
        Returns entries as MemoryEntry for compatibility.
        """
        query_lower = query.lower()
        results: list[MemoryEntry] = []
        for key, value in self._store.items():
            if query_lower in key.lower() or query_lower in value.lower():
                relevance = 1.0 if query_lower in key.lower() else 0.7
                results.append(
                    MemoryEntry(
                        id=key,
                        content=value,
                        metadata=self._meta.get(key, {}),
                        relevance=relevance,
                    )
                )
        results.sort(key=lambda e: e.relevance, reverse=True)
        return results[:limit]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Returns all working memory as context (query is ignored for selection)."""
        max_chars = max_tokens * _CHARS_PER_TOKEN
        lines = ["[Working Memory]"]
        total_chars = len(lines[0]) + 2
        for key, value in self._store.items():
            line = f"- {key}: {value}"
            if total_chars + len(line) + 2 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 2
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        value = self._store.get(entry_id)
        if value is None:
            return None
        return MemoryEntry(
            id=entry_id,
            content=value,
            metadata=self._meta.get(entry_id, {}),
        )

    async def delete(self, entry_id: str) -> bool:
        if entry_id in self._store:
            del self._store[entry_id]
            self._meta.pop(entry_id, None)
            return True
        return False

    async def clear(self) -> None:
        self._store.clear()
        self._meta.clear()

    async def count(self) -> int:
        return len(self._store)
