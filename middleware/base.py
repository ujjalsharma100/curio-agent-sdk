"""
Middleware base class and pipeline for intercepting LLM and tool calls.

Middleware provides a composable way to add cross-cutting concerns like
logging, cost tracking, rate limiting, and retry logic.

Hooks (HookRegistry) are emitted at the same lifecycle points when provided,
so observability can be implemented as hook consumers instead of middleware.
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, LLMStreamChunk
    from curio_agent_sdk.core.hooks import HookRegistry, HookContext

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

    async def on_llm_stream_chunk(
        self,
        request: LLMRequest,
        chunk: LLMStreamChunk,
    ) -> LLMStreamChunk | None:
        """
        Called for each chunk in a streaming LLM call.

        Return the (possibly modified) chunk, or None to drop it.
        """
        return chunk

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
    When hook_registry is provided, emits llm.call.before / llm.call.after / llm.call.error.
    """

    def __init__(
        self,
        middleware: list[Middleware],
        hook_registry: HookRegistry | None = None,
    ):
        self.middleware = list(middleware)
        self.hook_registry = hook_registry

    async def run_before_llm(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMRequest:
        """Run all before_llm_call hooks in order; emit llm.call.before if hook_registry set."""
        if self.hook_registry:
            from curio_agent_sdk.core.hooks import HookContext, LLM_CALL_BEFORE
            ctx = HookContext(
                event=LLM_CALL_BEFORE,
                data={"request": request},
                run_id=run_id or "",
                agent_id=agent_id or "",
            )
            await self.hook_registry.emit(LLM_CALL_BEFORE, ctx)
            if ctx.cancelled:
                raise RuntimeError("LLM call cancelled by hook")
            request = ctx.data.get("request", request)
        for mw in self.middleware:
            try:
                request = await mw.before_llm_call(request)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.before_llm_call failed: {e}")
                raise
        return request

    async def run_after_llm(
        self,
        request: LLMRequest,
        response: LLMResponse,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """Run all after_llm_call hooks in order; emit llm.call.after if hook_registry set."""
        for mw in self.middleware:
            try:
                response = await mw.after_llm_call(request, response)
            except Exception as e:
                logger.error(f"Middleware {mw.__class__.__name__}.after_llm_call failed: {e}")
                raise
        if self.hook_registry:
            from curio_agent_sdk.core.hooks import HookContext, LLM_CALL_AFTER
            ctx = HookContext(
                event=LLM_CALL_AFTER,
                data={"request": request, "response": response},
                run_id=run_id or "",
                agent_id=agent_id or "",
            )
            await self.hook_registry.emit(LLM_CALL_AFTER, ctx)
            response = ctx.data.get("response", response)
        return response

    async def run_stream_chunk(
        self,
        request: LLMRequest,
        chunk: LLMStreamChunk,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMStreamChunk | None:
        """
        Run per-chunk stream hooks in order for streaming LLM calls.

        Middleware can modify or drop chunks (by returning None).
        """
        for mw in self.middleware:
            try:
                chunk = await mw.on_llm_stream_chunk(request, chunk)
                if chunk is None:
                    return None
            except Exception as e:
                logger.error(
                    f"Middleware {mw.__class__.__name__}.on_llm_stream_chunk failed: {e}"
                )
                raise
        return chunk

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

    async def run_on_error(
        self,
        error: Exception,
        context: dict[str, Any],
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> Exception | None:
        """Run all on_error hooks; emit llm.call.error if hook_registry set. If any returns None, error is suppressed."""
        if self.hook_registry:
            from curio_agent_sdk.core.hooks import HookContext, LLM_CALL_ERROR
            ctx = HookContext(
                event=LLM_CALL_ERROR,
                data={**context, "error": str(error), "exception": error},
                run_id=run_id or "",
                agent_id=agent_id or "",
            )
            await self.hook_registry.emit(LLM_CALL_ERROR, ctx)
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
        """LLM call with middleware hooks and lifecycle hooks."""
        request = await self._pipeline.run_before_llm(request, run_id=run_id, agent_id=agent_id)
        try:
            response = await self._inner.call(request, run_id=run_id, agent_id=agent_id)
        except Exception as e:
            result = await self._pipeline.run_on_error(
                e,
                {"phase": "llm_call", "request": request},
                run_id=run_id,
                agent_id=agent_id,
            )
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
        response = await self._pipeline.run_after_llm(
            request, response, run_id=run_id, agent_id=agent_id
        )
        return response

    async def stream(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Streaming LLM call with before/after hooks and per-chunk middleware.

        - Runs before_llm_call hooks and llm.call.before hooks once at start
        - Runs on_llm_stream_chunk for each chunk (middleware can modify/drop)
        - Aggregates a synthetic LLMResponse at the end and runs after_llm_call
        """
        from curio_agent_sdk.models.llm import LLMResponse, Message, TokenUsage

        # Apply before-call middleware and hooks
        request = await self._pipeline.run_before_llm(
            request,
            run_id=run_id,
            agent_id=agent_id,
        )

        # Aggregate stream into a synthetic response for after_llm_call
        text_parts: list[str] = []
        tool_calls: list[Any] = []
        total_usage = TokenUsage()
        finish_reason: str | None = None

        try:
            async for chunk in self._inner.stream(
                request,
                run_id=run_id,
                agent_id=agent_id,
            ):
                # Per-chunk middleware (can modify or drop chunks)
                processed = await self._pipeline.run_stream_chunk(
                    request,
                    chunk,
                    run_id=run_id,
                    agent_id=agent_id,
                )
                if processed is None:
                    continue

                # Aggregate basic information for synthetic response
                if processed.type == "text_delta" and processed.text:
                    text_parts.append(processed.text)
                if processed.type == "tool_call_end" and processed.tool_call:
                    tool_calls.append(processed.tool_call)
                if processed.type in ("usage", "done") and processed.usage:
                    total_usage.input_tokens += processed.usage.input_tokens
                    total_usage.output_tokens += processed.usage.output_tokens
                    total_usage.cache_read_tokens += processed.usage.cache_read_tokens
                    total_usage.cache_write_tokens += processed.usage.cache_write_tokens
                if processed.type == "done" and processed.finish_reason:
                    finish_reason = processed.finish_reason

                yield processed
        except Exception as e:
            # Stream-level error handling and hooks
            result = await self._pipeline.run_on_error(
                e,
                {"phase": "llm_stream", "request": request},
                run_id=run_id,
                agent_id=agent_id,
            )
            if result is None:
                # Error suppressed – end the stream silently
                return
            raise result
        else:
            # Run after-call middleware and hooks with a synthetic response
            message = Message.assistant("".join(text_parts) if text_parts else "", tool_calls=tool_calls or None)
            response = LLMResponse(
                message=message,
                usage=total_usage,
                model=request.model or "",
                provider=request.provider or "",
                finish_reason=finish_reason or "stream",
            )
            await self._pipeline.run_after_llm(
                request,
                response,
                run_id=run_id,
                agent_id=agent_id,
            )
