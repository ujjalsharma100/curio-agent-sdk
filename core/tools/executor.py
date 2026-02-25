"""
Async tool executor that handles tool calls from LLM responses.

Manages execution of tool calls with proper error handling,
result formatting, and message construction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.models.llm import Message, ToolCall
from curio_agent_sdk.exceptions import ToolError, ToolNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of executing a single tool call."""
    tool_call_id: str
    tool_name: str
    result: Any
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def content(self) -> str:
        """Get the result as a string for the LLM."""
        if self.error:
            return f"Error: {self.error}"
        if isinstance(self.result, str):
            return self.result
        try:
            return json.dumps(self.result, indent=2, default=str)
        except (TypeError, ValueError):
            return str(self.result)

    def to_message(self) -> Message:
        """Convert to a tool result message for the conversation."""
        return Message.tool_result(
            tool_call_id=self.tool_call_id,
            content=self.content,
            name=self.tool_name,
        )


class ToolExecutor:
    """
    Executes tool calls from LLM responses.

    Handles:
    - Looking up tools in the registry
    - Executing with proper argument passing
    - Error handling and result formatting
    - Constructing tool result messages
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: The ToolCall from the LLM response.

        Returns:
            ToolResult with the execution result or error.
        """
        try:
            tool = self.registry.get(tool_call.name)
            result = await tool.execute(**tool_call.arguments)

            logger.info(f"Tool '{tool_call.name}' executed successfully")

            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
            )

        except ToolNotFoundError as e:
            logger.error(f"Tool not found: {tool_call.name}")
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=str(e),
            )
        except ToolError as e:
            logger.error(f"Tool error for '{tool_call.name}': {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error executing '{tool_call.name}': {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=f"Unexpected error: {e}",
            )

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """
        Execute multiple tool calls sequentially.

        Args:
            tool_calls: List of ToolCall objects.

        Returns:
            List of ToolResult objects in same order.
        """
        results = []
        for tc in tool_calls:
            result = await self.execute(tc)
            results.append(result)
        return results

    async def execute_to_messages(self, tool_calls: list[ToolCall]) -> list[Message]:
        """
        Execute tool calls and return as tool result messages.

        This is the most common usage pattern in agent loops:
        LLM returns tool_calls -> execute -> append result messages -> continue.
        """
        results = await self.execute_all(tool_calls)
        return [r.to_message() for r in results]
