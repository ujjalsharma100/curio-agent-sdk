"""
Integration tests: Agent + Tools (Phase 17 §21.1)

Validates end-to-end tool calling through the agent using MockLLM.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Test tools ────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def formatter(text: str, style: str = "upper") -> str:
    """Format text."""
    if style == "upper":
        return text.upper()
    return text.lower()


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_calls_single_tool():
    """LLM requests tool → tool executed → result fed back → final response."""
    mock = MockLLM()
    mock.add_tool_call_response("calculator", {"expression": "2+2"})
    mock.add_text_response("The answer is 4.")

    agent = Agent(tools=[calculator], system_prompt="You are a calculator.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("What is 2+2?")

    assert result.status == "completed"
    assert "4" in result.output
    assert len(harness.tool_calls) == 1
    assert harness.tool_calls[0][0] == "calculator"
    assert harness.tool_calls[0][1] == {"expression": "2+2"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_calls_multiple_tools():
    """Sequential tool calls across iterations."""
    mock = MockLLM()
    mock.add_tool_call_response("search", {"query": "weather"})
    mock.add_tool_call_response("formatter", {"text": "sunny", "style": "upper"})
    mock.add_text_response("The weather is SUNNY.")

    agent = Agent(tools=[search, formatter], system_prompt="Help user.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("What is the weather?")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 2
    assert harness.tool_calls[0][0] == "search"
    assert harness.tool_calls[1][0] == "formatter"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_parallel_tool_calls():
    """Parallel tool execution via multiple tool calls in one response."""
    from curio_agent_sdk.models.llm import ToolCall as TC, Message, TokenUsage, LLMResponse

    tc1 = TC(id="call_1", name="calculator", arguments={"expression": "1+1"})
    tc2 = TC(id="call_2", name="search", arguments={"query": "news"})
    parallel_response = LLMResponse(
        message=Message.assistant("", tool_calls=[tc1, tc2]),
        usage=TokenUsage(input_tokens=20, output_tokens=10),
        model="mock-model",
        provider="mock",
        finish_reason="tool_use",
    )

    mock = MockLLM()
    mock.add_response(parallel_response)
    mock.add_text_response("Got both results.")

    agent = Agent(tools=[calculator, search], system_prompt="Help.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Do both things")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 2
    tool_names = {tc[0] for tc in harness.tool_calls}
    assert tool_names == {"calculator", "search"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_tool_error_recovery():
    """Tool error → error message → LLM handles gracefully."""
    @tool
    def failing_tool(x: str) -> str:
        """A tool that fails."""
        raise ValueError("Something went wrong")

    mock = MockLLM()
    mock.add_tool_call_response("failing_tool", {"x": "test"})
    mock.add_text_response("The tool failed, but I can help anyway.")

    agent = Agent(tools=[failing_tool], system_prompt="Handle errors.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Use the tool")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_tool_chain():
    """Tool A output → Tool B input (multi-step chain)."""
    mock = MockLLM()
    mock.add_tool_call_response("search", {"query": "python"})
    mock.add_tool_call_response("formatter", {"text": "Results for: python", "style": "upper"})
    mock.add_text_response("RESULTS FOR: PYTHON")

    agent = Agent(tools=[search, formatter], system_prompt="Chain tools.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Search and format")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 2
    assert harness.tool_calls[0][0] == "search"
    assert harness.tool_calls[1][0] == "formatter"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_no_tool_calls():
    """Simple text response with no tool calls."""
    mock = MockLLM()
    mock.add_text_response("Hello! I can help you.")

    agent = Agent(tools=[calculator], system_prompt="Helpful.", llm=mock)
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Hello")

    assert result.status == "completed"
    assert "Hello" in result.output
    assert len(harness.tool_calls) == 0
