"""
Testing utilities for the Curio Agent SDK.

Provides mock LLM clients, test harnesses, evaluation suites, and
regression detection for deterministic agent testing.

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
from curio_agent_sdk.testing.toolkit import ToolTestKit
from curio_agent_sdk.testing.integration import MultiAgentTestHarness
from curio_agent_sdk.testing.eval import (
    AgentEvalSuite,
    EvalDataset,
    EvalCase,
    EvalResult,
    EvalSuiteResult,
    exact_match,
    contains_match,
    tool_call_match,
    token_efficiency,
)
from curio_agent_sdk.testing.regression import (
    RegressionDetector,
    RegressionReport,
)

__all__ = [
    "MockLLM",
    "AgentTestHarness",
    "ToolTestKit",
    "MultiAgentTestHarness",
    "text_response",
    "tool_call_response",
    # Eval suite
    "AgentEvalSuite",
    "EvalDataset",
    "EvalCase",
    "EvalResult",
    "EvalSuiteResult",
    "exact_match",
    "contains_match",
    "tool_call_match",
    "token_efficiency",
    # Regression
    "RegressionDetector",
    "RegressionReport",
]
