"""
Performance tests: Middleware Overhead (Phase 19)

Validates that middleware pipeline and hook emission have acceptable overhead.
"""

import asyncio
import time
import pytest

from curio_agent_sdk.middleware.base import Middleware, MiddlewarePipeline
from curio_agent_sdk.core.events.hooks import HookRegistry, HookContext
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    TokenUsage,
)


# ── Helpers ───────────────────────────────────────────────────────────────


class NoOpMiddleware(Middleware):
    """A middleware that does nothing (measures pure overhead)."""

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        return response


class CountingMiddleware(Middleware):
    """A middleware that counts invocations."""

    def __init__(self):
        self.before_count = 0
        self.after_count = 0

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        self.before_count += 1
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        self.after_count += 1
        return response


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[Message.user("test")])


def _make_response() -> LLMResponse:
    return LLMResponse(
        message=Message.assistant("ok"),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="test",
        provider="test",
        finish_reason="stop",
    )


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.asyncio
async def test_middleware_pipeline_overhead():
    """1000 middleware pipeline passes with 5 middleware complete in < 3s."""
    middleware_list = [NoOpMiddleware() for _ in range(5)]
    pipeline = MiddlewarePipeline(middleware=middleware_list)

    request = _make_request()
    response = _make_response()

    start = time.monotonic()
    for _ in range(1000):
        req = await pipeline.run_before_llm(request)
        resp = await pipeline.run_after_llm(request, response)
    elapsed = time.monotonic() - start

    assert elapsed < 3.0, f"1000 pipeline passes took {elapsed:.2f}s (limit: 3s)"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_hook_emit_overhead():
    """10000 hook emissions complete in < 3s."""
    registry = HookRegistry()

    call_count = 0

    async def noop_handler(ctx: HookContext) -> HookContext:
        nonlocal call_count
        call_count += 1
        return ctx

    # Register 5 handlers on the same event
    for _ in range(5):
        registry.on("test.event", noop_handler)

    start = time.monotonic()
    for i in range(10000):
        ctx = HookContext(event="test.event", data={"i": i})
        await registry.emit("test.event", ctx)
    elapsed = time.monotonic() - start

    assert call_count == 50000  # 10000 emissions * 5 handlers
    assert elapsed < 3.0, f"10000 hook emissions took {elapsed:.2f}s (limit: 3s)"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_middleware_counting_accuracy():
    """Middleware invocation counts are accurate at scale."""
    counter = CountingMiddleware()
    pipeline = MiddlewarePipeline(middleware=[counter])

    request = _make_request()
    response = _make_response()

    for _ in range(5000):
        await pipeline.run_before_llm(request)
        await pipeline.run_after_llm(request, response)

    assert counter.before_count == 5000
    assert counter.after_count == 5000
