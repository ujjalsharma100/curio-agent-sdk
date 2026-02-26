"""
Testing utilities for the Curio Agent SDK.

Provides mock LLM clients and test harnesses for deterministic agent testing.

Example:
    from curio_agent_sdk.testing import MockLLM, AgentTestHarness, text_response

    mock = MockLLM()
    mock.add_text_response("The answer is 42.")

    agent = Agent(tools=[...], system_prompt="...")
    harness = AgentTestHarness(agent, llm=mock)
    result = harness.run_sync("What is the answer?")

    assert result.status == "completed"
    assert "42" in result.output
    assert mock.call_count == 1
"""

from curio_agent_sdk.testing.mock_llm import MockLLM, text_response, tool_call_response
from curio_agent_sdk.testing.harness import AgentTestHarness

__all__ = [
    "MockLLM",
    "AgentTestHarness",
    "text_response",
    "tool_call_response",
]
