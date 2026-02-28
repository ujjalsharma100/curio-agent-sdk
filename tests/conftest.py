"""
Root conftest.py — Shared fixtures for all tests.
"""

import pytest
from datetime import datetime

from curio_agent_sdk.models.llm import (
    Message,
    ToolCall,
    TokenUsage,
    ContentBlock,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    ToolSchema,
)
from curio_agent_sdk.models.agent import (
    AgentRun,
    AgentRunResult,
    AgentRunStatus,
    AgentRunEvent,
    AgentLLMUsage,
)
from curio_agent_sdk.models.events import AgentEvent, EventType, StreamEvent


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_messages():
    """A typical short conversation: system + user + assistant."""
    return [
        Message.system("You are a helpful assistant."),
        Message.user("Hello!"),
        Message.assistant("Hi there! How can I help you?"),
    ]


@pytest.fixture
def sample_tool_call():
    """A single ToolCall instance."""
    return ToolCall(id="call_123", name="calculator", arguments={"expression": "2+2"})


@pytest.fixture
def sample_tool_schema():
    """A simple ToolSchema."""
    return ToolSchema(
        name="calculator",
        description="Evaluate a math expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
    )


# ---------------------------------------------------------------------------
# LLM request / response factories
# ---------------------------------------------------------------------------

@pytest.fixture
def text_response_factory():
    """Factory that creates a text-only LLMResponse."""

    def _make(text: str = "Hello!", model: str = "mock-model", provider: str = "mock"):
        return LLMResponse(
            message=Message.assistant(text),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model=model,
            provider=provider,
            finish_reason="stop",
        )

    return _make


@pytest.fixture
def tool_call_response_factory():
    """Factory that creates an LLMResponse containing a tool call."""

    def _make(
        tool_name: str = "calculator",
        arguments: dict | None = None,
        model: str = "mock-model",
        provider: str = "mock",
    ):
        tc = ToolCall(id="call_1", name=tool_name, arguments=arguments or {})
        return LLMResponse(
            message=Message.assistant("", tool_calls=[tc]),
            usage=TokenUsage(input_tokens=15, output_tokens=8),
            model=model,
            provider=provider,
            finish_reason="tool_use",
        )

    return _make


# ---------------------------------------------------------------------------
# Event collector (for hook tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def event_collector():
    """Returns a list that can be used as a hook handler to collect events."""
    collected: list[dict] = []

    def handler(ctx):
        collected.append({"event": ctx.event, "data": dict(ctx.data)})

    handler.collected = collected  # type: ignore[attr-defined]
    return handler
