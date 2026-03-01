"""
E2E tests: Structured Agent (Phase 18 §22.8)

Validates agent returning parsed Pydantic models and structured JSON.
"""

import json
import pytest

from pydantic import BaseModel

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.testing.mock_llm import MockLLM


# ── Models ────────────────────────────────────────────────────────────────


class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    keywords: list[str] = []


class TaskBreakdown(BaseModel):
    title: str
    steps: list[str]
    estimated_hours: float


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_structured_output():
    """Agent returns a parsed Pydantic model as structured output."""
    mock = MockLLM()
    mock.add_text_response(json.dumps({
        "text": "I love this product! It's amazing.",
        "sentiment": "positive",
        "confidence": 0.95,
        "keywords": ["love", "amazing", "product"],
    }))

    agent = Agent(
        system_prompt="You are a sentiment analysis agent. Return structured results.",
        llm=mock,
    )
    result = await agent.arun(
        "Analyze the sentiment: 'I love this product! It's amazing.'",
        response_format=SentimentResult,
    )

    assert result.status == "completed"
    assert result.parsed_output is not None
    assert isinstance(result.parsed_output, SentimentResult)
    assert result.parsed_output.sentiment == "positive"
    assert result.parsed_output.confidence == 0.95
    assert "love" in result.parsed_output.keywords
    assert result.parsed_output.text == "I love this product! It's amazing."


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_json_schema_output():
    """Agent returns structured JSON matching a Pydantic schema."""
    mock = MockLLM()
    mock.add_text_response(json.dumps({
        "title": "Build a REST API",
        "steps": [
            "Define endpoints and data models",
            "Set up FastAPI project",
            "Implement CRUD operations",
            "Add authentication",
            "Write tests",
        ],
        "estimated_hours": 16.5,
    }))

    agent = Agent(
        system_prompt="You are a project planner. Break down tasks into steps.",
        llm=mock,
    )
    result = await agent.arun(
        "Break down the task: Build a REST API",
        response_format=TaskBreakdown,
    )

    assert result.status == "completed"
    assert result.parsed_output is not None
    assert isinstance(result.parsed_output, TaskBreakdown)
    assert result.parsed_output.title == "Build a REST API"
    assert len(result.parsed_output.steps) == 5
    assert result.parsed_output.estimated_hours == 16.5
    # Verify it can also be serialized back
    d = result.parsed_output.model_dump()
    assert isinstance(d, dict)
    assert "title" in d
    assert "steps" in d
