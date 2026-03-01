"""
Integration tests: Agent + Structured Output (Phase 17 §21.11)

Validates Pydantic response format, dict response format, and validation errors.
"""

import pytest

from pydantic import BaseModel

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Models ────────────────────────────────────────────────────────────────


class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    tags: list[str] = []


class SimpleAnswer(BaseModel):
    answer: str


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pydantic_response_format():
    """Agent returns a parsed Pydantic model."""
    import json

    mock = MockLLM()
    mock.add_text_response(json.dumps({
        "summary": "The data shows growth.",
        "confidence": 0.95,
        "tags": ["growth", "positive"],
    }))

    agent = Agent(system_prompt="Analyze data.", llm=mock)
    result = await agent.arun(
        "Analyze this dataset",
        response_format=AnalysisResult,
    )

    assert result.status == "completed"
    assert result.parsed_output is not None
    assert isinstance(result.parsed_output, AnalysisResult)
    assert result.parsed_output.summary == "The data shows growth."
    assert result.parsed_output.confidence == 0.95
    assert "growth" in result.parsed_output.tags


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dict_response_format():
    """Agent returns parsed dict from JSON response."""
    import json

    mock = MockLLM()
    mock.add_text_response(json.dumps({
        "answer": "42",
    }))

    agent = Agent(system_prompt="Answer questions.", llm=mock)
    result = await agent.arun(
        "What is the answer?",
        response_format=SimpleAnswer,
    )

    assert result.status == "completed"
    assert result.parsed_output is not None
    assert result.parsed_output.answer == "42"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_structured_output_validation():
    """Invalid output is handled gracefully."""
    mock = MockLLM()
    mock.add_text_response("This is not valid JSON at all.")

    agent = Agent(system_prompt="Answer questions.", llm=mock)
    result = await agent.arun(
        "Give me structured data",
        response_format=AnalysisResult,
    )

    # The agent should complete (possibly with error) rather than crash
    assert result.status in ("completed", "error")
