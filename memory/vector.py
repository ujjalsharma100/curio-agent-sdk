"""
Vector memory - long-term semantic memory using embeddings.

Stores memories as vectors for similarity-based retrieval.
Supports OpenAI embeddings out of the box, with a pluggable
embedding function for custom providers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Awaitable

from curio_agent_sdk.core.component import Component
from curio_agent_sdk.memory.base import Memory, MemoryEntry

logger = logging.getLogger(__name__)

# Approximate chars per token
_CHARS_PER_TOKEN = 4


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _default_embedding_fn(texts: list[str]) -> list[list[float]]:
    """
    Default embedding function using OpenAI's API.

    Requires the `openai` package to be installed and
    OPENAI_API_KEY to be set.
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "VectorMemory requires the 'openai' package for default embeddings. "
            "Install with: pip install openai\n"
            "Or provide a custom embedding_fn."
        )

    client = openai.AsyncOpenAI()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


class VectorMemory(Memory, Component):
    """
    Long-term semantic memory using vector embeddings.

    Stores content alongside embedding vectors for similarity-based
    retrieval. Uses cosine similarity for search.

    By default uses OpenAI's text-embedding-3-small model. Provide
    a custom `embedding_fn` for other embedding providers.

    Example:
        # Default (OpenAI embeddings)
        memory = VectorMemory()

        # Custom embeddings
        async def my_embeddings(texts):
            return [my_model.encode(t) for t in texts]

        memory = VectorMemory(embedding_fn=my_embeddings)

        await memory.add("Quantum computing uses qubits")
        results = await memory.search("What are qubits?")
    """

    def __init__(
        self,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        embedding_model: str = "text-embedding-3-small",
        persist_path: Path | str | None = None,
    ):
        self.embedding_fn = embedding_fn or _default_embedding_fn
        self.embedding_model = embedding_model
        self._persist_path = Path(persist_path).expanduser().resolve() if persist_path else None
        self._entries: list[MemoryEntry] = []
        self._vectors: list[list[float]] = []
        self._index: dict[str, int] = {}  # entry_id -> index

    # ── Component lifecycle (persistence: load/save when persist_path is set) ──

    async def startup(self) -> None:
        """Load index from disk if persist_path is set (no-op otherwise)."""
        if not self._persist_path:
            return
        path = self._persist_path
        if not path.suffix:
            path = path.with_suffix(".json")
        if not path.exists():
            return
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
            data = json.loads(raw)
            entries_data = data.get("entries", [])
            vectors_data = data.get("vectors", [])
            self._entries = [MemoryEntry.from_dict(e) for e in entries_data]
            self._vectors = [list(v) for v in vectors_data]
            self._index = {e.id: i for i, e in enumerate(self._entries)}
            if len(self._entries) != len(self._vectors):
                logger.warning(
                    "VectorMemory persist file has %d entries and %d vectors; resetting.",
                    len(self._entries),
                    len(self._vectors),
                )
                self._entries.clear()
                self._vectors.clear()
                self._index.clear()
        except Exception as e:
            logger.warning("Failed to load VectorMemory from %s: %s", path, e)

    async def shutdown(self) -> None:
        """Save index to disk if persist_path is set (no-op otherwise)."""
        if not self._persist_path or not self._entries:
            return
        path = self._persist_path
        if not path.suffix:
            path = path.with_suffix(".json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [e.to_dict() for e in self._entries],
                "vectors": self._vectors,
            }
            await asyncio.to_thread(
                path.write_text,
                json.dumps(data, ensure_ascii=False, indent=0),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save VectorMemory to %s: %s", path, e)

    async def health_check(self) -> bool:
        """Return True if the memory is ready to use."""
        return True

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        return await self.embedding_fn(texts)

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
        )

        # Generate embedding
        vectors = await self._embed([content])
        vector = vectors[0]

        idx = len(self._entries)
        self._entries.append(entry)
        self._vectors.append(vector)
        self._index[entry.id] = idx

        return entry.id

    async def add_batch(self, items: list[tuple[str, dict[str, Any] | None]]) -> list[str]:
        """
        Add multiple entries in a single batch (more efficient for embeddings).

        Args:
            items: List of (content, metadata) tuples.

        Returns:
            List of entry IDs.
        """
        if not items:
            return []

        contents = [content for content, _ in items]
        vectors = await self._embed(contents)

        ids = []
        for (content, metadata), vector in zip(items, vectors):
            entry = MemoryEntry(
                content=content,
                metadata=metadata or {},
            )
            idx = len(self._entries)
            self._entries.append(entry)
            self._vectors.append(vector)
            self._index[entry.id] = idx
            ids.append(entry.id)

        return ids

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """
        Search memories by semantic similarity to the query.

        Args:
            query: The search query.
            limit: Maximum results to return.

        Returns:
            List of MemoryEntry objects sorted by relevance (cosine similarity).
        """
        if not self._entries:
            return []

        # Embed the query
        query_vectors = await self._embed([query])
        query_vector = query_vectors[0]

        # Compute similarities
        scored: list[tuple[float, MemoryEntry]] = []
        for entry, vector in zip(self._entries, self._vectors):
            similarity = _cosine_similarity(query_vector, vector)
            result = MemoryEntry(
                id=entry.id,
                content=entry.content,
                metadata=entry.metadata,
                relevance=similarity,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
            )
            scored.append((similarity, result))

        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get semantically relevant context for prompt inclusion.
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN

        # Search with a generous limit, then trim to fit
        results = await self.search(query, limit=20)

        selected: list[MemoryEntry] = []
        total_chars = 0

        for entry in results:
            entry_chars = len(entry.content) + 30
            if total_chars + entry_chars > max_chars:
                break
            if entry.relevance < 0.1:  # Skip very low relevance
                continue
            selected.append(entry)
            total_chars += entry_chars

        if not selected:
            return ""

        lines = ["[Relevant Memories]"]
        for entry in selected:
            lines.append(f"- {entry.content} (relevance: {entry.relevance:.2f})")
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        idx = self._index.get(entry_id)
        if idx is not None and idx < len(self._entries):
            return self._entries[idx]
        return None

    async def delete(self, entry_id: str) -> bool:
        idx = self._index.get(entry_id)
        if idx is None:
            return False

        # Remove by swapping with last (to avoid reindexing everything)
        last_idx = len(self._entries) - 1
        if idx != last_idx:
            # Swap with last
            last_entry = self._entries[last_idx]
            self._entries[idx] = last_entry
            self._vectors[idx] = self._vectors[last_idx]
            self._index[last_entry.id] = idx

        self._entries.pop()
        self._vectors.pop()
        del self._index[entry_id]
        return True

    async def clear(self) -> None:
        self._entries.clear()
        self._vectors.clear()
        self._index.clear()

    async def count(self) -> int:
        return len(self._entries)
