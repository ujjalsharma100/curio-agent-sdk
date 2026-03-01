"""
Unit tests for curio_agent_sdk.memory.policies — decay, importance, summarization.
"""

from datetime import datetime, timedelta, timezone

import pytest

from curio_agent_sdk.memory.base import MemoryEntry
from curio_agent_sdk.memory.policies import (
    importance_score,
    decay_score,
    combined_relevance,
    summarize_old_memories,
)
from curio_agent_sdk.memory.conversation import ConversationMemory


@pytest.mark.unit
class TestMemoryPolicies:
    def test_importance_policy(self):
        entry = MemoryEntry(content="x", metadata={"importance": 0.9})
        assert importance_score(entry) == 0.9
        entry2 = MemoryEntry(content="y", metadata={})
        assert importance_score(entry2, default=0.5) == 0.5
        entry3 = MemoryEntry(content="z", metadata={"importance": "0.7"})
        assert importance_score(entry3) == 0.7

    def test_importance_invalid_metadata(self):
        entry = MemoryEntry(content="x", metadata={"importance": "invalid"})
        assert importance_score(entry, default=0.5) == 0.5

    def test_decay_policy(self):
        now = datetime.now(timezone.utc)
        entry_new = MemoryEntry(content="new", metadata={})
        entry_new.created_at = now
        entry_old = MemoryEntry(content="old", metadata={})
        entry_old.created_at = now - timedelta(days=60)
        assert decay_score(entry_new, now=now) == 1.0
        # After ~2 half-lives (60 days if half_life=30), score ~0.25
        old_score = decay_score(entry_old, now=now, half_life_days=30.0)
        assert 0 < old_score < 0.5

    def test_combined_policies(self):
        entry = MemoryEntry(
            content="x",
            metadata={"importance": 0.8},
        )
        entry.created_at = datetime.now(timezone.utc)
        score = combined_relevance(
            0.5,
            entry,
            importance_weight=0.3,
            decay_weight=0.2,
            half_life_days=30.0,
        )
        assert 0 <= score <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
class TestSummarizeOldMemories:
    async def test_summarize_old_memories_skips_when_few_entries(self):
        mem = ConversationMemory(max_entries=100)
        for i in range(5):
            await mem.add(f"entry {i}")

        async def summarizer(contents):
            return "Summary of " + str(len(contents))

        n = await summarize_old_memories(
            mem,
            summarizer,
            max_entries=100,
            min_entries_to_compress=20,
        )
        assert n == 0

    async def test_summarize_old_memories_compresses(self):
        mem = ConversationMemory(max_entries=100)
        for i in range(25):
            await mem.add(f"memory entry {i}")

        async def summarizer(contents):
            return "Compressed: " + str(len(contents)) + " items"

        n = await summarize_old_memories(
            mem,
            summarizer,
            max_entries=50,
            min_entries_to_compress=20,
        )
        assert n == 20
        # Old entries replaced by one summary
        results = await mem.search("Compressed", limit=5)
        assert len(results) >= 1
        assert "Compressed" in results[0].content
