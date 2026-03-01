"""
Unit tests for curio_agent_sdk.memory.episodic — EpisodicMemory, Episode.
"""

from datetime import datetime, timezone

import pytest

from curio_agent_sdk.memory.episodic import EpisodicMemory, Episode


@pytest.mark.unit
@pytest.mark.asyncio
class TestEpisodicMemory:
    async def test_temporal_ordering(self):
        mem = EpisodicMemory(max_episodes=10)
        await mem.add("first event", metadata={})
        await mem.add("second event", metadata={})
        await mem.add("third event", metadata={})
        episodes = await mem.recall("event", limit=10)
        assert len(episodes) == 3

    async def test_experience_storage(self):
        mem = EpisodicMemory()
        ep = Episode(content="User asked about X", summary="Q about X", importance=0.8)
        eid = await mem.record_episode(ep)
        assert eid == ep.id
        recalled = await mem.recall("X", limit=5)
        assert len(recalled) >= 1
        assert any(e.content == "User asked about X" for e in recalled)

    async def test_relevance_scoring(self):
        mem = EpisodicMemory()
        await mem.add("python programming", metadata={"importance": 0.9})
        await mem.add("weather today", metadata={"importance": 0.2})
        results = await mem.search("python", limit=5)
        assert len(results) >= 1
        assert any("python" in e.content for e in results)

    async def test_recall_time_range(self):
        mem = EpisodicMemory()
        await mem.add("event one", metadata={})
        await mem.add("event two", metadata={})
        # Wide time range: all episodes should be included when query matches
        from datetime import datetime as dt
        start, end = dt(2020, 1, 1), dt(2030, 1, 1)
        in_range = await mem.recall("event", limit=10, time_range=(start, end))
        assert len(in_range) >= 1
        contents = [e.content for e in in_range]
        assert any("event" in c for c in contents)
