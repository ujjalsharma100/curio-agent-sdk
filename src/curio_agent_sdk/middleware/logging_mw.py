"""
Logging middleware for structured logging of all LLM and tool operations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger("curio_agent_sdk.middleware.logging")


class LoggingMiddleware(Middleware):
    """
    Logs all LLM calls and tool calls with structured information.

    Configurable log level and logger name.

    .. deprecated::
        Consider using :class:`~curio_agent_sdk.middleware.consumers.LoggingConsumer`
        as a hook-based alternative. ``LoggingConsumer`` attaches to
        ``HookRegistry`` events and supports automatic trace-id/span-id
        correlation when OpenTelemetry is available.

    Example:
        agent = Agent(
            middleware=[LoggingMiddleware(level=logging.DEBUG)],
            ...
        )
    """

    def __init__(
        self,
        level: int = logging.INFO,
        logger_name: str = "curio_agent_sdk.middleware.logging",
    ):
        self.level = level
        self.log = logging.getLogger(logger_name)
        self._call_start: float = 0.0

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        self._call_start = time.monotonic()
        self.log.log(self.level,
            "LLM call started | model=%s provider=%s tier=%s messages=%d tools=%d",
            request.model or "auto",
            request.provider or "auto",
            request.tier or "default",
            len(request.messages),
            len(request.tools) if request.tools else 0,
        )
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        elapsed = (time.monotonic() - self._call_start) * 1000
        self.log.log(self.level,
            "LLM call completed | model=%s provider=%s finish=%s "
            "input_tokens=%d output_tokens=%d latency=%.0fms",
            response.model,
            response.provider,
            response.finish_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
            elapsed,
        )
        return response

    async def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        self.log.log(self.level, "Tool call started | tool=%s args=%s", tool_name, args)
        return tool_name, args

    async def after_tool_call(self, tool_name: str, args: dict[str, Any], result: Any) -> Any:
        result_preview = str(result)[:200] if result is not None else "None"
        self.log.log(self.level, "Tool call completed | tool=%s result=%s", tool_name, result_preview)
        return result

    async def on_error(self, error: Exception, context: dict[str, Any]) -> Exception | None:
        self.log.error("Error in %s: %s", context.get("phase", "unknown"), error)
        return error
