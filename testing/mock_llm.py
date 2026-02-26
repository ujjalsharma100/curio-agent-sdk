"""
Mock LLM client for deterministic agent testing.
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import Any, AsyncIterator

from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
    TokenUsage,
)


def text_response(text: str, model: str = "mock-model") -> LLMResponse:
    """Create a simple text LLMResponse."""
    return LLMResponse(
        message=Message.assistant(text),
        usage=TokenUsage(input_tokens=len(text) // 4, output_tokens=len(text) // 4),
        model=model,
        provider="mock",
        finish_reason="stop",
    )


def tool_call_response(
    name: str,
    arguments: dict[str, Any],
    text: str = "",
    model: str = "mock-model",
) -> LLMResponse:
    """Create a tool call LLMResponse."""
    tc = ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)
    return LLMResponse(
        message=Message.assistant(text, tool_calls=[tc]),
        usage=TokenUsage(input_tokens=50, output_tokens=30),
        model=model,
        provider="mock",
        finish_reason="tool_use",
    )


class MockLLM:
    """
    Mock LLM client that returns pre-configured responses.

    Drop-in replacement for LLMClient in tests.

    Example:
        mock = MockLLM()
        mock.add_text_response("Hello!")
        mock.add_tool_call_response([ToolCall(id="1", name="search", arguments={"q": "test"})])
        mock.add_text_response("Here are the results.")

        # Pass to Agent or AgentTestHarness
        agent = Agent(llm=mock, ...)
    """

    def __init__(self) -> None:
        self._responses: deque[LLMResponse] = deque()
        self.calls: list[LLMRequest] = []
        self.call_count: int = 0

    def add_response(self, response: LLMResponse) -> None:
        """Queue a response to return on the next call."""
        self._responses.append(response)

    def add_text_response(self, text: str, model: str = "mock-model") -> None:
        """Queue a simple text response."""
        self._responses.append(text_response(text, model))

    def add_tool_call_response(
        self,
        tool_calls: list[ToolCall],
        text: str = "",
        model: str = "mock-model",
    ) -> None:
        """Queue a tool call response."""
        self._responses.append(LLMResponse(
            message=Message.assistant(text, tool_calls=tool_calls),
            usage=TokenUsage(input_tokens=50, output_tokens=30),
            model=model,
            provider="mock",
            finish_reason="tool_use",
        ))

    async def call(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """Return the next queued response, or a default stop response."""
        self.calls.append(request)
        self.call_count += 1

        if self._responses:
            return self._responses.popleft()

        return text_response("I'm done.", "mock-model")

    async def stream(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream text chunks from the next queued response."""
        response = await self.call(request, run_id, agent_id)
        text = response.message.text
        if text:
            for i in range(0, len(text), 10):
                yield LLMStreamChunk(type="text_delta", text=text[i : i + 10])
        yield LLMStreamChunk(
            type="done",
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    @property
    def request_messages(self) -> list[list[Message]]:
        """All message lists from recorded calls."""
        return [req.messages for req in self.calls]

    # Proxy unknown attributes (e.g. router) to avoid AttributeError
    def __getattr__(self, name: str) -> Any:
        return None
