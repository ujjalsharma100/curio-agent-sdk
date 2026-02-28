"""
Memory policies — decay, importance, and summarization helpers.

Optional utilities for advanced memory strategies:
- Importance scoring (from metadata or constant)
- Time-based decay for relevance
- Summarization of old memories to compress context
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from curio_agent_sdk.memory.base import Memory, MemoryEntry


def importance_score(entry: MemoryEntry, default: float = 0.5) -> float:
    """
    Return importance score for an entry (0.0–1.0).

    Uses entry.metadata["importance"] if present, else default.
    """
    v = entry.metadata.get("importance")
    if v is None:
        return default
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def decay_score(
    entry: MemoryEntry,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    """
    Time-based decay multiplier (0.0–1.0). Newer = higher.

    Uses entry.created_at. half_life_days: days after which score halves.
    """
    now = now or datetime.now(timezone.utc)
    created = entry.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    delta = (now - created).total_seconds() / 86400.0  # days
    if delta <= 0:
        return 1.0
    return math.exp(-delta * (math.log(2) / half_life_days))


def combined_relevance(
    base_relevance: float,
    entry: MemoryEntry,
    *,
    importance_weight: float = 0.3,
    decay_weight: float = 0.2,
    half_life_days: float = 30.0,
    now: datetime | None = None,
) -> float:
    """
    Combine search relevance with importance and time decay.

    Final score = (1 - i - d) * base + i * importance + d * decay,
    so base_relevance dominates by default.
    """
    imp = importance_score(entry)
    dec = decay_score(entry, now=now, half_life_days=half_life_days)
    return (
        (1.0 - importance_weight - decay_weight) * base_relevance
        + importance_weight * imp
        + decay_weight * dec
    )


async def summarize_old_memories(
    memory: Memory,
    summarizer_fn: Callable[[list[str]], Awaitable[str]],
    *,
    max_entries: int = 100,
    min_entries_to_compress: int = 20,
    namespace: str | None = None,
) -> int:
    """
    Compress old memories by summarizing and replacing with one summary entry.

    Fetches entries (e.g. oldest or by search), summarizes them with
    summarizer_fn(list_of_contents) -> summary_str, deletes the old entries,
    and adds one new entry with the summary. Returns the number of entries
    that were replaced (0 if nothing was done).

    Requires memory to support search and delete. summarizer_fn can be an
    LLM-based summarizer. If memory does not support listing by age, this
    may be a no-op; EpisodicMemory and ConversationMemory order by recency.
    """
    # Generic path: get many entries via search with a broad query, then
    # take the oldest N. Not all backends support "oldest" so we document
    # that this is best-effort.
    results = await memory.search("", limit=max_entries)
    if len(results) < min_entries_to_compress:
        return 0
    # Assume results might be in relevance order; for "oldest" we'd need
    # backend support. Here we take the tail as "old" and summarize.
    to_compress = results[-min_entries_to_compress:]
    contents = [e.content for e in to_compress]
    summary = await summarizer_fn(contents)
    meta: dict[str, Any] = {"type": "compressed_summary", "count": len(to_compress)}
    if namespace:
        meta["namespace"] = namespace
    await memory.add(summary, metadata=meta)
    for e in to_compress:
        await memory.delete(e.id)
    return len(to_compress)
