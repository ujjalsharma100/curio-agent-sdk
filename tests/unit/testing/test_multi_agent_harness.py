"""
Unit tests for MultiAgentTestHarness (Phase 16 — Testing Utilities).
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.testing.integration import MultiAgentTestHarness


@pytest.mark.unit
def test_multi_agent_harness():
    """MultiAgentTestHarness: add_agent, get_agent."""
    a1 = Agent(system_prompt="Agent 1", tools=[])
    a2 = Agent(system_prompt="Agent 2", tools=[])
    harness = MultiAgentTestHarness(agents={"planner": a1, "worker": a2})
    assert harness.get_agent("planner") is a1
    assert harness.get_agent("worker") is a2
    with pytest.raises(KeyError, match="Unknown agent"):
        harness.get_agent("unknown")
    harness.add_agent("extra", Agent(system_prompt="Extra", tools=[]))
    assert "extra" in harness.agents
