"""
Verify tests — run key SDK flows with mocks and assert expected behavior.

Use these to confirm the SDK works without live API keys:
  pytest tests/verify -v
  pytest -m verify -v
"""

import tempfile
from pathlib import Path

import pytest

from curio_agent_sdk import Agent, use_run_logger
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.harness import AgentTestHarness
from curio_agent_sdk.testing.mock_llm import MockLLM


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool agent + run logger
# ---------------------------------------------------------------------------


@pytest.mark.verify
@pytest.mark.asyncio
async def test_tool_agent_with_run_logger_produces_log_file():
    """Run a tool-calling agent with use_run_logger; assert log file exists and contains expected sections."""
    mock = MockLLM()
    mock.add_tool_call_response("calculator", {"expression": "10 + 5"})
    mock.add_text_response("The result is 15.")

    with tempfile.TemporaryDirectory() as tmp:
        builder = (
            Agent.builder()
            .system_prompt("You are a math assistant. Use the calculator.")
            .tools([calculator])
            .llm(mock)
        )
        logger = use_run_logger(builder, base_name="verify-sdk", output_dir=tmp)
        agent = builder.build()
        harness = AgentTestHarness(agent, llm=mock)
        result = await harness.run("What is 10 + 5?")

        assert result.status == "completed"
        assert "15" in result.output
        assert len(harness.tool_calls) == 1
        assert harness.tool_calls[0][0] == "calculator"

        log_path = logger.get_log_path()
        assert log_path is not None
        assert Path(log_path).exists()
        content = Path(log_path).read_text()

        assert "[AGENT RUN START]" in content
        assert "What is 10 + 5?" in content
        assert "[LLM REQUEST]" in content
        assert "[LLM RESPONSE]" in content
        assert "[TOOL CALL START]" in content
        assert "calculator" in content
        assert "[TOOL CALL END]" in content
        assert "15" in content
        assert "[AGENT RUN END]" in content
        assert "toolCalls count: 1" in content


@pytest.mark.verify
@pytest.mark.asyncio
async def test_tool_agent_schema_and_tool_execution():
    """Verify tool schema is accepted and tool is executed with correct args."""
    mock = MockLLM()
    mock.add_tool_call_response("calculator", {"expression": "7 * 8"})
    mock.add_text_response("7 times 8 equals 56.")

    agent = Agent(
        system_prompt="You are a math assistant.",
        tools=[calculator],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Compute 7 * 8")

    assert result.status == "completed"
    assert "56" in result.output
    assert len(harness.tool_calls) == 1
    name, args = harness.tool_calls[0]
    assert name == "calculator"
    assert args.get("expression") == "7 * 8"
