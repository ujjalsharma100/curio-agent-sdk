"""
Episodic memory — experiences with temporal context.

Stores experiences as episodes (what happened, when) for recall by
query and optional time range.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from curio_agent_sdk.memory.base import Memory, MemoryEntry

_CHARS_PER_TOKEN = 4


@dataclass
class Episode:
    """
    A single episodic memory: what happened, when, and optional metadata.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0–1.0 for decay/ranking

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        ep = cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            metadata=data.get("metadata", {}),
            importance=float(data.get("importance", 0.5)),
        )
        if data.get("created_at"):
            ep.created_at = datetime.fromisoformat(data["created_at"])
        return ep


class EpisodicMemory(Memory):
    """
    Remembers experiences as episodes with temporal context.

    Supports record_episode, recall by query and optional time_range.
    Implements Memory via add (creates episode) and search/get_context
    (recall by relevance and time).

    Example:
        memory = EpisodicMemory()
        await memory.record_episode(Episode(content="User asked about X", summary="Q about X"))
        episodes = await memory.recall("X", time_range=(start, end))
    """

    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self._episodes: list[Episode] = []
        self._index: dict[str, int] = {}  # id -> index

    async def record_episode(self, episode: Episode) -> str:
        """Record an episode. Returns episode id."""
        if not episode.id:
            episode.id = uuid.uuid4().hex[:12]
        if episode.created_at is None:
            episode.created_at = datetime.now()
        idx = len(self._episodes)
        self._episodes.append(episode)
        self._index[episode.id] = idx
        # Evict oldest if over limit
        while len(self._episodes) > self.max_episodes:
            old = self._episodes.pop(0)
            self._index.pop(old.id, None)
            # Rebuild index for remaining
            self._index = {e.id: i for i, e in enumerate(self._episodes)}
        return episode.id

    async def recall(
        self,
        query: str,
        limit: int = 10,
        time_range: tuple[datetime | None, datetime | None] | None = None,
    ) -> list[Episode]:
        """
        Recall episodes by keyword relevance and optional time range.

        time_range: (start, end) — only episodes with created_at in [start, end].
        """
        query_lower = query.lower()
        terms = query_lower.split()
        start_ts, end_ts = time_range or (None, None)

        scored: list[tuple[float, Episode]] = []
        for ep in self._episodes:
            if start_ts is not None and ep.created_at < start_ts:
                continue
            if end_ts is not None and ep.created_at > end_ts:
                continue
            text = f"{ep.content} {ep.summary}".lower()
            hits = sum(1 for t in terms if t in text)
            relevance = (hits / max(len(terms), 1)) * 0.7 + ep.importance * 0.3
            if hits > 0 or relevance > 0:
                scored.append((relevance, ep))
        scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add content as an episode. Metadata can include 'summary', 'importance'."""
        meta = metadata or {}
        ep = Episode(
            content=content,
            summary=meta.get("summary", ""),
            metadata=meta,
            importance=float(meta.get("importance", 0.5)),
        )
        return await self.record_episode(ep)

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search episodes by query; returns MemoryEntry for compatibility."""
        episodes = await self.recall(query, limit=limit)
        return [
            MemoryEntry(
                id=ep.id,
                content=ep.content or ep.summary,
                metadata=ep.metadata,
                relevance=0.0,  # not computed per-entry here
                created_at=ep.created_at,
            )
            for ep in episodes
        ]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get recalled episodes formatted for prompt inclusion."""
        max_chars = max_tokens * _CHARS_PER_TOKEN
        episodes = await self.recall(query, limit=20)
        lines = ["[Episodic Memory]"]
        total_chars = len(lines[0]) + 2
        for ep in episodes:
            line = f"- [{ep.created_at.isoformat()}] {ep.content or ep.summary}"
            if total_chars + len(line) + 2 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 2
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        idx = self._index.get(entry_id)
        if idx is None:
            return None
        ep = self._episodes[idx]
        return MemoryEntry(
            id=ep.id,
            content=ep.content or ep.summary,
            metadata=ep.metadata,
            created_at=ep.created_at,
        )

    async def delete(self, entry_id: str) -> bool:
        idx = self._index.get(entry_id)
        if idx is None:
            return False
        self._episodes.pop(idx)
        del self._index[entry_id]
        self._index = {e.id: i for i, e in enumerate(self._episodes)}
        return True

    async def clear(self) -> None:
        self._episodes.clear()
        self._index.clear()

    async def count(self) -> int:
        return len(self._episodes)
