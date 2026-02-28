"""
Test harness for running agents with deterministic mock LLMs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.executor import ToolResult
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.toolkit import ToolTestKit


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

    def __init__(
        self,
        agent: Agent,
        llm: MockLLM | None = None,
        tool_kit: ToolTestKit | None = None,
    ) -> None:
        self.agent = agent
        self.mock_llm = llm or MockLLM()
        self.tool_kit = tool_kit
        self.result: AgentRunResult | None = None
        self._tool_calls: list[tuple[str, dict[str, Any]]] = []

        # Inject mock LLM
        self.agent.llm = self.mock_llm  # type: ignore[assignment]
        self.agent._wire_loop()

        # Attach registry to tool kit (for schema validation) if provided
        if self.tool_kit is not None and getattr(self.agent, "registry", None) is not None:
            self.tool_kit._attach_registry(self.agent.registry)  # type: ignore[arg-type]

        # Wrap tool executor to track calls
        original_execute = self.agent.executor.execute

        async def tracking_execute(tool_call, *args, **kwargs):
            tool_name = getattr(tool_call, "name", "")
            args_dict = dict(getattr(tool_call, "arguments", {}) or {})

            # Always record high-level call info for simple assertions
            self._tool_calls.append((tool_name, args_dict))

            # If a ToolTestKit is attached, let it validate/record and optionally mock
            if self.tool_kit is not None:
                mock = self.tool_kit._get_mock(tool_name)
                if mock is not None:
                    # Schema validation + record happen through _record_call
                    if isinstance(mock.side_effect, Exception):
                        # Simulate tool error
                            # This will raise inside _record_call if schema is invalid
                        self.tool_kit._record_call(tool_name, args_dict, result=None, error=str(mock.side_effect))
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            tool_name=tool_name,
                            result=None,
                            error=str(mock.side_effect),
                        )
                    try:
                        if callable(mock.side_effect):
                            result_value = mock.side_effect(args_dict)
                        else:
                            result_value = mock.returns
                        self.tool_kit._record_call(tool_name, args_dict, result=result_value, error=None)
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            tool_name=tool_name,
                            result=result_value,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        self.tool_kit._record_call(tool_name, args_dict, result=None, error=str(exc))
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            tool_name=tool_name,
                            result=None,
                            error=str(exc),
                        )

                # No mock configured: just record the call (and optionally validate)
                self.tool_kit._record_call(tool_name, args_dict)

            # Fall back to the original executor for real execution
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

    def set_llm(self, llm: Any) -> None:
        """
        Replace the underlying LLM client used by the agent.

        Useful for swapping in a ReplayLLMClient for deterministic tests.
        """
        self.mock_llm = llm  # type: ignore[assignment]
        self.agent.llm = llm  # type: ignore[assignment]
        self.agent._wire_loop()

    async def run_conversation(
        self,
        inputs: Sequence[str],
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[AgentRunResult]:
        """
        Run a multi-turn conversation against the agent with a persistent session.

        Args:
            inputs: Sequence of user messages for each turn.
            session_id: Optional explicit session ID. If omitted, a unique
                test session ID is generated.
            **kwargs: Extra keyword arguments forwarded to Agent.arun()
                (e.g., context, max_iterations, response_format).

        Returns:
            List of AgentRunResult objects, one per turn.
        """
        if not inputs:
            return []

        if session_id is None:
            import uuid
            session_id = f"test-session-{uuid.uuid4().hex[:8]}"

        # Reset per-run tracking
        self._tool_calls.clear()
        if self.tool_kit is not None:
            self.tool_kit.clear_calls()

        results: list[AgentRunResult] = []
        for text in inputs:
            result = await self.agent.arun(text, session_id=session_id, **kwargs)
            self.result = result
            results.append(result)
        return results

    def run_conversation_sync(
        self,
        inputs: Sequence[str],
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[AgentRunResult]:
        """
        Synchronous wrapper around run_conversation().
        """
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.run_conversation(inputs, session_id=session_id, **kwargs),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.run_conversation(inputs, session_id=session_id, **kwargs))

    @property
    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """List of (tool_name, args) for all tools called."""
        return list(self._tool_calls)

    @property
    def llm_calls(self) -> int:
        """Number of LLM calls made."""
        return getattr(self.mock_llm, "call_count", 0)
