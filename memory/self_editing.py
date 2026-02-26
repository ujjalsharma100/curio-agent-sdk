"""
Self-editing memory (MemGPT/Letta style).

Core memory is always in context; archival memory is searchable.
Agent manages memory via tools: core_memory_read/write/replace,
archival_memory_search/insert.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.memory.key_value import KeyValueMemory

if TYPE_CHECKING:
    from curio_agent_sdk.core.tools.tool import Tool

_CHARS_PER_TOKEN = 4


class SelfEditingMemory(Memory):
    """
    Memory the agent can explicitly read/write/edit.

    - Core memory: short text that is ALWAYS injected into context.
    - Archival memory: searchable store (default KeyValueMemory).

    get_tools() returns: core_memory_read, core_memory_write, core_memory_replace,
    archival_memory_search, archival_memory_insert.

    Example:
        memory = SelfEditingMemory()
        tools = memory.get_tools()
        agent = Agent(tools=[*my_tools, *tools], memory_manager=MemoryManager(memory=memory, ...))
    """

    def __init__(self, archival: Memory | None = None, max_core_chars: int = 2000):
        self._core_memory_text = ""
        self.max_core_chars = max_core_chars
        self._archival = archival or KeyValueMemory()

    @property
    def core_memory_text(self) -> str:
        return self._core_memory_text

    def get_tools(self) -> list[Tool]:
        """Tools for the agent to manage its own memory."""
        from curio_agent_sdk.core.tools.tool import tool as tool_decorator

        mem = self

        @tool_decorator
        async def core_memory_read() -> str:
            """Read the core memory (always-in-context facts about the user and the agent). Returns the current core memory text."""
            return mem._core_memory_text or "(Core memory is empty.)"

        @tool_decorator
        async def core_memory_write(addition: str) -> str:
            """Append to the core memory. Use for adding new facts. Does not replace existing content."""
            mem._core_memory_text = (mem._core_memory_text + "\n" + addition).strip()
            if mem.max_core_chars and len(mem._core_memory_text) > mem.max_core_chars:
                mem._core_memory_text = mem._core_memory_text[-mem.max_core_chars:]
            return "Core memory updated."

        @tool_decorator
        async def core_memory_replace(old_string: str, new_string: str) -> str:
            """Replace a substring in the core memory. Use when correcting or updating a specific fact. Pass the exact text to replace and the new text."""
            if old_string in mem._core_memory_text:
                mem._core_memory_text = mem._core_memory_text.replace(old_string, new_string, 1)
                return "Core memory updated."
            return "The given substring was not found in core memory."

        @tool_decorator
        async def archival_memory_search(query: str, limit: int = 5) -> str:
            """Search the archival (long-term) memory for relevant information. Returns matching entries."""
            results = await mem._archival.search(query, limit=limit)
            if not results:
                return "No relevant entries in archival memory."
            lines = [f"- {e.content}" for e in results]
            return "\n".join(lines)

        @tool_decorator
        async def archival_memory_insert(content: str) -> str:
            """Insert a new entry into archival memory. Use for storing important information for later retrieval."""
            entry_id = await mem._archival.add(content, metadata={"type": "archival"})
            return f"Inserted into archival memory (id: {entry_id})."

        return [
            core_memory_read,
            core_memory_write,
            core_memory_replace,
            archival_memory_search,
            archival_memory_insert,
        ]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Core memory is ALWAYS in context; optionally append relevant archival results."""
        parts = []
        if self._core_memory_text:
            parts.append("[Core Memory]\n" + self._core_memory_text)
        # Add some archival context if there's room
        archival_budget = max(0, (max_tokens * _CHARS_PER_TOKEN) - len(self._core_memory_text) - 100)
        if archival_budget > 100:
            results = await self._archival.search(query, limit=5)
            if results:
                lines = ["[Relevant Archival Memory]"]
                total = len(lines[0]) + 2
                for e in results:
                    line = f"- {e.content}"
                    if total + len(line) + 2 > archival_budget:
                        break
                    lines.append(line)
                    total += len(line) + 2
                if len(lines) > 1:
                    parts.append("\n".join(lines))
        return "\n\n".join(parts) if parts else ""

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add to archival memory (core is managed via tools)."""
        return await self._archival.add(content, metadata=metadata)

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search archival memory."""
        return await self._archival.search(query, limit=limit)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        return await self._archival.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        return await self._archival.delete(entry_id)

    async def clear(self) -> None:
        self._core_memory_text = ""
        await self._archival.clear()

    async def count(self) -> int:
        return await self._archival.count()
