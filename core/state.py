"""
Agent state - the mutable context passed through the agent loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from curio_agent_sdk.models.llm import Message, ToolSchema
from curio_agent_sdk.core.tools.tool import Tool


@dataclass
class AgentState:
    """
    Mutable state container passed through the agent loop.

    This holds everything the loop needs to operate:
    - The conversation message history
    - Available tools
    - Iteration counter
    - Cancellation signal
    - Metrics accumulators
    """
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    tool_schemas: list[ToolSchema] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 25
    metadata: dict[str, Any] = field(default_factory=dict)

    # Metrics
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Control
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _done: bool = False
    _last_finish_reason: str = ""

    def cancel(self):
        """Signal the agent to stop after the current step."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def is_done(self) -> bool:
        return self._done

    def mark_done(self):
        """Mark the loop as complete."""
        self._done = True

    def add_message(self, message: Message):
        """Append a message to the conversation history."""
        self.messages.append(message)

    def add_messages(self, messages: list[Message]):
        """Append multiple messages."""
        self.messages.extend(messages)

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record an LLM call in the metrics."""
        self.total_llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_tool_calls(self, count: int = 1):
        """Record tool call(s) in the metrics."""
        self.total_tool_calls += count

    @property
    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def assistant_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]
