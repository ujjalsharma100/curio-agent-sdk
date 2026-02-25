"""
Composite memory - combines multiple memory types.

Routes adds to all sub-memories and merges search results
across all types, sorted by relevance.
"""

from __future__ import annotations

import logging
from typing import Any

from curio_agent_sdk.memory.base import Memory, MemoryEntry

logger = logging.getLogger(__name__)

# Approximate chars per token
_CHARS_PER_TOKEN = 4


class CompositeMemory(Memory):
    """
    Combines multiple memory types into a unified interface.

    Adds go to all sub-memories. Searches merge results across
    all sub-memories and return the top results by relevance.

    Example:
        from curio_agent_sdk.memory import (
            CompositeMemory, ConversationMemory,
            VectorMemory, KeyValueMemory,
        )

        memory = CompositeMemory({
            "conversation": ConversationMemory(max_entries=50),
            "knowledge": VectorMemory(),
            "facts": KeyValueMemory(),
        })

        # Adds go to all sub-memories
        await memory.add("User's name is Alice")

        # Or add to a specific sub-memory
        await memory.get_memory("facts").set("user_name", "Alice")

        # Search across all
        results = await memory.search("user name")
    """

    def __init__(self, memories: dict[str, Memory]):
        """
        Args:
            memories: Dict mapping name -> Memory instance.
        """
        self.memories = dict(memories)

    def get_memory(self, name: str) -> Memory:
        """Get a specific sub-memory by name."""
        return self.memories[name]

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Add to all sub-memories.

        The metadata dict can include a "memory_targets" list to
        restrict which sub-memories receive the entry. If not
        specified, all sub-memories receive it.

        Returns the ID from the first sub-memory.
        """
        meta = metadata or {}
        targets = meta.get("memory_targets")

        first_id = ""
        for name, memory in self.memories.items():
            if targets and name not in targets:
                continue
            entry_id = await memory.add(content, metadata=meta)
            if not first_id:
                first_id = entry_id

        return first_id

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Search across all sub-memories and merge results by relevance.
        """
        all_results: list[MemoryEntry] = []

        for name, memory in self.memories.items():
            results = await memory.search(query, limit=limit)
            for entry in results:
                # Tag with source memory name
                entry.metadata = {**entry.metadata, "_source_memory": name}
                all_results.append(entry)

        # Sort by relevance across all sources
        all_results.sort(key=lambda e: e.relevance, reverse=True)

        # Deduplicate by content (different memories may have same entry)
        seen_content: set[str] = set()
        deduped: list[MemoryEntry] = []
        for entry in all_results:
            if entry.content not in seen_content:
                seen_content.add(entry.content)
                deduped.append(entry)

        return deduped[:limit]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get context from all sub-memories, each with a budget share.
        """
        if not self.memories:
            return ""

        # Give each memory an equal token budget
        per_memory_tokens = max_tokens // len(self.memories)
        sections: list[str] = []

        for name, memory in self.memories.items():
            context = await memory.get_context(query, max_tokens=per_memory_tokens)
            if context:
                sections.append(context)

        return "\n\n".join(sections)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Search all sub-memories for the entry."""
        for memory in self.memories.values():
            entry = await memory.get(entry_id)
            if entry is not None:
                return entry
        return None

    async def delete(self, entry_id: str) -> bool:
        """Delete from all sub-memories."""
        deleted = False
        for memory in self.memories.values():
            if await memory.delete(entry_id):
                deleted = True
        return deleted

    async def clear(self) -> None:
        """Clear all sub-memories."""
        for memory in self.memories.values():
            await memory.clear()

    async def count(self) -> int:
        """Total entries across all sub-memories."""
        total = 0
        for memory in self.memories.values():
            total += await memory.count()
        return total
