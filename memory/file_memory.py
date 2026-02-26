"""
File-based memory (Claude Code style).

Memory stored as files on disk. Persists across restarts.
Supports namespacing via subdirectories.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from curio_agent_sdk.core.component import Component
from curio_agent_sdk.memory.base import Memory, MemoryEntry

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4


def _expand_path(path: Path) -> Path:
    return path.expanduser().resolve()


class FileMemory(Memory, Component):
    """
    Memory stored as files on disk.

    Each entry is a file in memory_dir. Supports namespace via
    subdirectories (memory_dir / namespace). Persists across restarts.

    Example:
        memory = FileMemory(memory_dir=Path("~/.agent/memory"))
        await memory.add("User prefers dark mode")
        context = await memory.get_context("preferences", max_tokens=1000)
    """

    def __init__(
        self,
        memory_dir: Path | str | None = None,
        namespace: str | None = None,
    ):
        if memory_dir is None:
            memory_dir = Path("~/.agent/memory")
        self._base_dir = Path(memory_dir) if isinstance(memory_dir, str) else memory_dir
        self._namespace = namespace
        self._index_file = None  # base dir or namespace dir / _index.json
        self._entries_index: dict[str, str] = {}  # id -> relative path (for get/delete)

    def _dir(self) -> Path:
        d = _expand_path(self._base_dir)
        if self._namespace:
            d = d / self._namespace
        return d

    async def startup(self) -> None:
        """Ensure memory directory exists."""
        d = self._dir()
        await asyncio.to_thread(d.mkdir, parents=True, exist_ok=True)
        await self._load_index()

    async def shutdown(self) -> None:
        """Persist index if needed (entries are already files)."""
        await self._save_index()

    async def health_check(self) -> bool:
        d = self._dir()
        return await asyncio.to_thread(d.exists)

    def _index_path(self) -> Path:
        return self._dir() / "_index.json"

    async def _load_index(self) -> None:
        p = self._index_path()
        if not p.exists():
            self._entries_index = {}
            return
        try:
            raw = await asyncio.to_thread(p.read_text, encoding="utf-8")
            data = json.loads(raw)
            self._entries_index = data.get("entries", {})
        except Exception as e:
            logger.warning("Failed to load memory index: %s", e)
            self._entries_index = {}

    async def _save_index(self) -> None:
        p = self._index_path()
        data = {"entries": self._entries_index}
        await asyncio.to_thread(p.write_text, json.dumps(data, indent=0), encoding="utf-8")

    def _safe_filename(self, id: str) -> str:
        safe = re.sub(r"[^\w\-.]", "_", id)
        return (safe or "entry") + ".json"

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Write content to a new file. Returns entry id."""
        entry_id = uuid.uuid4().hex[:12]
        meta = metadata or {}
        rel_path = self._safe_filename(entry_id)
        full_path = self._dir() / rel_path
        payload = {
            "id": entry_id,
            "content": content,
            "metadata": meta,
        }
        await asyncio.to_thread(
            full_path.write_text,
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        self._entries_index[entry_id] = rel_path
        await self._save_index()
        return entry_id

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search by reading all entry files and matching query against content."""
        d = self._dir()
        if not d.exists():
            return []
        query_lower = query.lower()
        terms = query_lower.split()
        results: list[tuple[float, MemoryEntry]] = []

        def _read_entries():
            entries = []
            for f in d.iterdir():
                if f.name.startswith("_") or f.suffix != ".json":
                    continue
                try:
                    raw = f.read_text(encoding="utf-8")
                    data = json.loads(raw)
                    entries.append((f, data))
                except Exception:
                    continue
            return entries

        files_data = await asyncio.to_thread(_read_entries)
        for f, data in files_data:
            content = data.get("content", "")
            content_lower = content.lower()
            hits = sum(1 for t in terms if t in content_lower)
            relevance = hits / max(len(terms), 1)
            if relevance > 0:
                entry = MemoryEntry(
                    id=data.get("id", f.stem),
                    content=content,
                    metadata=data.get("metadata", {}),
                    relevance=relevance,
                )
                results.append((relevance, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:limit]]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Read and return relevant files' content."""
        max_chars = max_tokens * _CHARS_PER_TOKEN
        results = await self.search(query, limit=20)
        if not results:
            # Include all entries that fit
            d = self._dir()
            if not d.exists():
                return ""

            def _read_all():
                out = []
                for f in sorted(d.iterdir()):
                    if f.name.startswith("_") or f.suffix != ".json":
                        continue
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        out.append(data.get("content", ""))
                    except Exception:
                        continue
                return out

            contents = await asyncio.to_thread(_read_all)
        else:
            contents = [e.content for e in results]
        lines = ["[File Memory]"]
        total = len(lines[0]) + 2
        for content in contents:
            line = f"- {content}"
            if total + len(line) + 2 > max_chars:
                break
            lines.append(line)
            total += len(line) + 2
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        rel = self._entries_index.get(entry_id)
        if not rel:
            return None
        full = self._dir() / rel
        if not full.exists():
            self._entries_index.pop(entry_id, None)
            return None
        try:
            raw = await asyncio.to_thread(full.read_text, encoding="utf-8")
            data = json.loads(raw)
            return MemoryEntry(
                id=data.get("id", entry_id),
                content=data.get("content", ""),
                metadata=data.get("metadata", {}),
            )
        except Exception:
            return None

    async def delete(self, entry_id: str) -> bool:
        rel = self._entries_index.get(entry_id)
        if not rel:
            return False
        full = self._dir() / rel
        try:
            await asyncio.to_thread(full.unlink, missing_ok=True)
        except Exception as e:
            logger.warning("Failed to delete memory file %s: %s", full, e)
        self._entries_index.pop(entry_id, None)
        await self._save_index()
        return True

    async def clear(self) -> None:
        d = self._dir()
        if not d.exists():
            return
        for f in d.iterdir():
            if f.name.startswith("_"):
                continue
            try:
                await asyncio.to_thread(f.unlink)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", f, e)
        self._entries_index.clear()
        await self._save_index()

    async def count(self) -> int:
        return len(self._entries_index)
