"""
E2E tests: Tool Agent (Phase 18 §22.2)

Validates agent calling tools, orchestrating multiple tools, and handling failures.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny, 72°F."


@tool
def translate(text: str, language: str) -> str:
    """Translate text to a language."""
    return f"[{language}] {text}"


@tool
def broken_tool(data: str) -> str:
    """A tool that always fails."""
    raise RuntimeError("Service unavailable")


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_calculator_agent():
    """Agent calls calculator tool to solve a math problem."""
    mock = MockLLM()
    mock.add_tool_call_response("calculator", {"expression": "15 * 7 + 3"})
    mock.add_text_response("The result of 15 × 7 + 3 is 108.")

    agent = Agent(
        system_prompt="You are a math assistant. Use the calculator tool.",
        tools=[calculator],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("What is 15 times 7 plus 3?")

    assert result.status == "completed"
    assert "108" in result.output
    assert len(harness.tool_calls) == 1
    assert harness.tool_calls[0][0] == "calculator"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_tool_agent():
    """Agent orchestrates multiple tools in sequence."""
    mock = MockLLM()
    mock.add_tool_call_response("weather", {"city": "San Francisco"})
    mock.add_tool_call_response("translate", {"text": "sunny, 72°F", "language": "Spanish"})
    mock.add_text_response("The weather in San Francisco is sunny, 72°F. In Spanish: [Spanish] sunny, 72°F.")

    agent = Agent(
        system_prompt="You are a helpful travel assistant.",
        tools=[weather, translate],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("What's the weather in SF? Translate it to Spanish.")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 2
    tool_names = [tc[0] for tc in harness.tool_calls]
    assert "weather" in tool_names
    assert "translate" in tool_names


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tool_error_handling():
    """Agent handles tool failures gracefully and responds to user."""
    mock = MockLLM()
    mock.add_tool_call_response("broken_tool", {"data": "test"})
    mock.add_text_response("I'm sorry, the service is currently unavailable. Let me try to help another way.")

    agent = Agent(
        system_prompt="You are a helpful assistant. Handle errors gracefully.",
        tools=[broken_tool, calculator],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Use the broken tool")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 1
    # Agent should still produce a response despite the tool error
    assert len(result.output) > 0
