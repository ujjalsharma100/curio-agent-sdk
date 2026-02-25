"""
Middleware base class and pipeline for intercepting LLM and tool calls.

Middleware provides a composable way to add cross-cutting concerns like
logging, cost tracking, rate limiting, and retry logic.
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """
    Abstract base class for middleware.

    Middleware intercepts LLM calls and tool calls, allowing you to
    add logging, cost tracking, rate limiting, retries, and more.

    All hooks have default no-op implementations so you only override
    the ones you need.

    Example:
        class MyMiddleware(Middleware):
            async def before_llm_call(self, request):
                print(f"Calling LLM with {len(request.messages)} messages")
                return request

            async def after_llm_call(self, request, response):
                print(f"LLM responded: {response.finish_reason}")
                return response
    """

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        """Called before each LLM call. Can modify the request."""
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        """Called after each LLM call. Can modify the response."""
        return response

    async def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Called before each tool call. Can modify tool name and args."""
        return tool_name, args

    async def after_tool_call(self, tool_name: str, args: dict[str, Any], result: Any) -> Any:
        """Called after each tool call. Can modify the result."""
        return result

    async def on_error(self, error: Exception, context: dict[str, Any]) -> Exception | None:
        """
        Called on errors. Return None to suppress the error, or return
        the (possibly modified) error to propagate it.
        """
        return error


class MiddlewarePipeline:
    """
    Runs a list of middleware in order for LLM and tool call hooks.

    The pipeline wraps an LLMClient to transparently intercept calls.
    """

    def __init__(self, middleware: list[Middleware]):
        self.middleware = list(middleware)

    async def run_before_llm(self, request: LLMRequest) -> LLMRequest:
        """Run all before_llm_call hooks in order."""
        for mw in self.middleware:
            try:
                request = await mw.before_llm_call(request)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.before_llm_call failed: {e}")
                raise
        return request

    async def run_after_llm(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        """Run all after_llm_call hooks in order."""
        for mw in self.middleware:
            try:
                response = await mw.after_llm_call(request, response)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.after_llm_call failed: {e}")
                raise
        return response

    async def run_before_tool(self, tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Run all before_tool_call hooks in order."""
        for mw in self.middleware:
            try:
                tool_name, args = await mw.before_tool_call(tool_name, args)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.before_tool_call failed: {e}")
                raise
        return tool_name, args

    async def run_after_tool(self, tool_name: str, args: dict[str, Any], result: Any) -> Any:
        """Run all after_tool_call hooks in order."""
        for mw in self.middleware:
            try:
                result = await mw.after_tool_call(tool_name, args, result)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.after_tool_call failed: {e}")
                raise
        return result

    async def run_on_error(self, error: Exception, context: dict[str, Any]) -> Exception | None:
        """Run all on_error hooks. If any returns None, error is suppressed."""
        for mw in self.middleware:
            try:
                error = await mw.on_error(error, context)
                if error is None:
                    return None
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.on_error failed: {e}")
        return error

    def wrap_llm_client(self, client: Any) -> _MiddlewareWrappedLLMClient:
        """Wrap an LLMClient with this middleware pipeline."""
        return _MiddlewareWrappedLLMClient(client, self)


class _MiddlewareWrappedLLMClient:
    """
    LLMClient wrapper that applies middleware before/after each call.

    Transparently replaces the LLMClient so loops don't need to know
    about middleware.
    """

    def __init__(self, inner: Any, pipeline: MiddlewarePipeline):
        self._inner = inner
        self._pipeline = pipeline

    # Proxy attributes to inner client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def call(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """LLM call with middleware hooks."""
        request = await self._pipeline.run_before_llm(request)
        try:
            response = await self._inner.call(request, run_id=run_id, agent_id=agent_id)
        except Exception as e:
            result = await self._pipeline.run_on_error(e, {"phase": "llm_call", "request": request})
            if result is None:
                # Error suppressed - return a minimal error response
                from curio_agent_sdk.models.llm import LLMResponse, Message, TokenUsage
                return LLMResponse(
                    message=Message.assistant(""),
                    usage=TokenUsage(),
                    model=request.model or "",
                    provider=request.provider or "",
                    finish_reason="error",
                    error=str(e),
                )
            raise result
        response = await self._pipeline.run_after_llm(request, response)
        return response

    async def stream(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streaming LLM call with before-hook (after-hook not applied to streams)."""
        request = await self._pipeline.run_before_llm(request)
        async for chunk in self._inner.stream(request, run_id=run_id, agent_id=agent_id):
            yield chunk
