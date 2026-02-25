"""
Core LLM data models: messages, tool calls, requests, and responses.

These models provide a provider-agnostic interface for interacting with LLMs.
All providers translate to/from these models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    """A tool/function call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ContentBlock:
    """A block of content within a message (text, image, tool use, etc.)."""
    type: Literal["text", "image_url", "tool_use", "tool_result"]
    text: str | None = None
    image_url: str | None = None
    tool_call: ToolCall | None = None
    tool_call_id: str | None = None


@dataclass
class Message:
    """
    A single message in a conversation.

    Supports system, user, assistant, and tool roles.
    Content can be a simple string or a list of content blocks for multimodal.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool-role messages: which call this responds to
    name: str | None = None  # Optional name for the message sender

    @property
    def text(self) -> str:
        """Get plain text content, joining content blocks if needed."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for block in self.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def system(content: str) -> Message:
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role="user", content=content)

    @staticmethod
    def assistant(content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return Message(role="assistant", content=content, tool_calls=tool_calls)

    @staticmethod
    def tool_result(tool_call_id: str, content: str, name: str | None = None) -> Message:
        return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolSchema:
    """
    Schema for a tool that can be passed to an LLM for native tool calling.
    Uses JSON Schema format for parameters.
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class TokenUsage:
    """Token usage for an LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMRequest:
    """
    A request to an LLM provider. Provider-agnostic.

    This is the primary interface for making LLM calls. The LLMClient
    translates this into provider-specific API calls.
    """
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", "required", or {"name": "..."}
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False
    response_format: dict | None = None  # e.g. {"type": "json_object"}
    stop: list[str] | None = None
    model: str | None = None  # Override model selection
    provider: str | None = None  # Override provider selection
    tier: str | None = None  # Use tiered routing
    metadata: dict[str, Any] = field(default_factory=dict)  # Pass-through metadata


@dataclass
class LLMResponse:
    """
    Response from an LLM provider. Provider-agnostic.
    """
    message: Message
    usage: TokenUsage
    model: str
    provider: str
    finish_reason: str  # "stop", "tool_use", "length", "error"
    latency_ms: int = 0
    raw_response: Any = None  # Provider-specific raw response for debugging
    error: str | None = None

    @property
    def content(self) -> str:
        """Shortcut to get the text content of the response."""
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Shortcut to get tool calls from the response."""
        return self.message.tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)


@dataclass
class LLMStreamChunk:
    """A single chunk from a streaming LLM response."""
    type: Literal["text_delta", "tool_call_start", "tool_call_delta", "tool_call_end", "usage", "done"]
    text: str | None = None
    tool_call: ToolCall | None = None  # For tool_call_start
    tool_call_id: str | None = None
    argument_delta: str | None = None  # For tool_call_delta
    usage: TokenUsage | None = None  # For usage/done chunks
    finish_reason: str | None = None
