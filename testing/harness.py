"""
Test harness for running agents with deterministic mock LLMs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.testing.mock_llm import MockLLM


class AgentTestHarness:
    """
    Test harness for running agents with a MockLLM.

    Injects the mock LLM into the agent, runs it, and exposes
    inspection properties for assertions in tests.

    Example:
        from curio_agent_sdk import Agent, tool
        from curio_agent_sdk.testing import MockLLM, AgentTestHarness, tool_call_response

        @tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        mock = MockLLM()
        mock.add_response(tool_call_response("greet", {"name": "Alice"}))
        mock.add_text_response("I greeted Alice!")

        agent = Agent(tools=[greet], system_prompt="Greet people.")
        harness = AgentTestHarness(agent, llm=mock)
        result = harness.run_sync("Greet Alice")

        assert result.status == "completed"
        assert "Alice" in result.output
        assert harness.llm_calls == 2
    """

    def __init__(self, agent: Agent, llm: MockLLM | None = None) -> None:
        self.agent = agent
        self.mock_llm = llm or MockLLM()
        self.result: AgentRunResult | None = None
        self._tool_calls: list[tuple[str, dict[str, Any]]] = []

        # Inject mock LLM
        self.agent.llm = self.mock_llm  # type: ignore[assignment]
        self.agent._wire_loop()

        # Wrap tool executor to track calls
        original_execute = self.agent.executor.execute

        async def tracking_execute(tool_call, *args, **kwargs):
            self._tool_calls.append((tool_call.name, tool_call.arguments))
            return await original_execute(tool_call, *args, **kwargs)

        self.agent.executor.execute = tracking_execute  # type: ignore[assignment]

    async def run(self, input: str, **kwargs) -> AgentRunResult:
        """Run the agent asynchronously and return the result."""
        self._tool_calls.clear()
        self.result = await self.agent.arun(input, **kwargs)
        return self.result

    def run_sync(self, input: str, **kwargs) -> AgentRunResult:
        """Run the agent synchronously."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.run(input, **kwargs))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.run(input, **kwargs))

    @property
    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """List of (tool_name, args) for all tools called."""
        return list(self._tool_calls)

    @property
    def llm_calls(self) -> int:
        """Number of LLM calls made."""
        return self.mock_llm.call_count
