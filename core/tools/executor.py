"""
Async tool executor that handles tool calls from LLM responses.

Manages execution of tool calls with proper error handling,
result formatting, caching, and parallel execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.models.llm import Message, ToolCall
from curio_agent_sdk.exceptions import ToolError, ToolNotFoundError

if TYPE_CHECKING:
    from curio_agent_sdk.core.human_input import HumanInputHandler

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry with TTL."""
    result: Any
    expires_at: float


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
    - Parallel execution via execute_parallel()
    - Result caching with configurable TTL
    - Human-in-the-loop confirmation
    """

    def __init__(
        self,
        registry: ToolRegistry,
        human_input: HumanInputHandler | None = None,
    ):
        self.registry = registry
        self.human_input = human_input
        self._cache: dict[str, _CacheEntry] = {}

    def _cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Generate a deterministic cache key for a tool call."""
        payload = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_cached(self, tool_name: str, arguments: dict[str, Any], ttl: float) -> Any | None:
        """Return cached result if valid, else None."""
        key = self._cache_key(tool_name, arguments)
        entry = self._cache.get(key)
        if entry is not None:
            if time.monotonic() < entry.expires_at:
                return entry.result
            del self._cache[key]
        return None

    def _set_cached(self, tool_name: str, arguments: dict[str, Any], result: Any, ttl: float):
        """Store a result in the cache."""
        key = self._cache_key(tool_name, arguments)
        self._cache[key] = _CacheEntry(result=result, expires_at=time.monotonic() + ttl)

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

            # Human-in-the-loop confirmation
            if tool.config.require_confirmation and self.human_input:
                confirmed = await self.human_input.confirm_tool_call(
                    tool_call.name, tool_call.arguments
                )
                if not confirmed:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=None,
                        error="Tool call denied by user",
                    )

            # Check cache
            if tool.config.cache_ttl is not None:
                cached = self._get_cached(tool_call.name, tool_call.arguments, tool.config.cache_ttl)
                if cached is not None:
                    logger.debug(f"Cache hit for tool '{tool_call.name}'")
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=cached,
                    )

            result = await tool.execute(**tool_call.arguments)

            # Store in cache if configured
            if tool.config.cache_ttl is not None:
                self._set_cached(tool_call.name, tool_call.arguments, result, tool.config.cache_ttl)

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

    async def execute_parallel(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel using asyncio.gather.

        Errors are captured per-tool (not raised), so all tool calls complete
        even if some fail.

        Args:
            tool_calls: List of ToolCall objects.

        Returns:
            List of ToolResult objects in same order as input.
        """
        if not tool_calls:
            return []
        if len(tool_calls) == 1:
            return [await self.execute(tool_calls[0])]

        results = await asyncio.gather(
            *(self.execute(tc) for tc in tool_calls),
            return_exceptions=True,
        )

        # Convert any unexpected exceptions into ToolResult errors
        final: list[ToolResult] = []
        for i, r in enumerate(results):
            if isinstance(r, ToolResult):
                final.append(r)
            elif isinstance(r, Exception):
                final.append(ToolResult(
                    tool_call_id=tool_calls[i].id,
                    tool_name=tool_calls[i].name,
                    result=None,
                    error=f"Parallel execution error: {r}",
                ))
            else:
                final.append(r)
        return final

    async def execute_to_messages(self, tool_calls: list[ToolCall]) -> list[Message]:
        """
        Execute tool calls sequentially and return as tool result messages.
        """
        results = await self.execute_all(tool_calls)
        return [r.to_message() for r in results]

    async def execute_parallel_to_messages(self, tool_calls: list[ToolCall]) -> list[Message]:
        """
        Execute tool calls in parallel and return as tool result messages.
        """
        results = await self.execute_parallel(tool_calls)
        return [r.to_message() for r in results]
