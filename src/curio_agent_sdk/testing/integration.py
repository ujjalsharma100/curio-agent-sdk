"""
Integration testing utilities for multi-agent scenarios.

Provides helpers for:
- Handoffs between agents
- Subagent spawning
- Asserting shared memory between agents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.models.agent import AgentRunResult


@dataclass
class MultiAgentTestHarness:
    """
    Utilities for multi-agent integration tests.

    Typical usage:

        harness = MultiAgentTestHarness({
            "planner": planner_agent,
            "worker": worker_agent,
        })

        # Test handoff from planner -> worker
        result = await harness.handoff(
            from_agent="planner",
            to_agent="worker",
            context="Please implement the plan.",
        )
    """

    agents: Dict[str, Agent]

    def add_agent(self, name: str, agent: Agent) -> None:
        """Register an additional agent by name."""
        self.agents[name] = agent

    def get_agent(self, name: str) -> Agent:
        """Retrieve an agent by name."""
        if name not in self.agents:
            raise KeyError(f"Unknown agent '{name}'. Known agents: {list(self.agents.keys())}")
        return self.agents[name]

    async def handoff(
        self,
        from_agent: str,
        to_agent: str,
        *,
        context: str,
        parent_messages: list | None = None,
        run_id: str | None = None,
    ) -> AgentRunResult:
        """
        Execute a handoff from one agent to another.

        This wraps Agent.handoff() to make multi-agent interaction tests
        concise and explicit.
        """
        source = self.get_agent(from_agent)
        target = self.get_agent(to_agent)
        return await source.handoff(
            target,
            context=context,
            parent_messages=parent_messages,
            run_id=run_id,
        )

    async def spawn_subagent(
        self,
        parent: str,
        config: str | Any,
        task: str,
        **kwargs: Any,
    ) -> AgentRunResult:
        """
        Spawn a subagent from a parent agent and run it on *task*.

        This calls Agent.spawn_subagent() under the hood, making it easy
        to exercise subagent configurations in tests.
        """
        agent = self.get_agent(parent)
        return await agent.spawn_subagent(config, task, **kwargs)

    def assert_shared_memory(self, a: str, b: str) -> None:
        """
        Assert that two agents share the same memory manager instance.

        Useful for verifying shared-memory configurations between parent
        and subagents or between collaborating agents.
        """
        agent_a = self.get_agent(a)
        agent_b = self.get_agent(b)
        mm_a = getattr(agent_a, "memory_manager_instance", None)
        mm_b = getattr(agent_b, "memory_manager_instance", None)

        if mm_a is None or mm_b is None:
            raise AssertionError(
                f"Expected agents '{a}' and '{b}' to have memory managers, "
                f"but got {mm_a!r} and {mm_b!r}."
            )
        if mm_a is not mm_b:
            raise AssertionError(
                f"Expected agents '{a}' and '{b}' to share the same memory manager "
                f"instance, but they differ: {mm_a!r} vs {mm_b!r}."
            )

