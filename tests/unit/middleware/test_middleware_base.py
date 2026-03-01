"""
Unit tests for Middleware ABC and MiddlewarePipeline.
"""

import abc
import pytest
from unittest.mock import AsyncMock

from curio_agent_sdk.middleware.base import Middleware, MiddlewarePipeline
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, Message, TokenUsage
from curio_agent_sdk.core.events import HookRegistry, HookContext, LLM_CALL_BEFORE, LLM_CALL_AFTER, LLM_CALL_ERROR


def _make_request():
    return LLMRequest(messages=[Message.user("hi")], model="gpt-4o", provider="openai")


def _make_response():
    return LLMResponse(
        message=Message.assistant("hello"),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        model="gpt-4o",
        provider="openai",
        finish_reason="stop",
    )


class ConcreteMiddleware(Middleware):
    """Concrete implementation for testing."""
    pass


@pytest.mark.unit
class TestMiddleware:
    def test_middleware_is_abstract(self):
        assert issubclass(Middleware, abc.ABC)

    def test_middleware_default_passthrough(self):
        mw = ConcreteMiddleware()
        req = _make_request()
        resp = _make_response()

        @pytest.mark.asyncio
        async def _run():
            r = await mw.before_llm_call(req)
            assert r is req
            r2 = await mw.after_llm_call(req, resp)
            assert r2 is resp
            tn, args = await mw.before_tool_call("x", {})
            assert tn == "x" and args == {}
            res = await mw.after_tool_call("x", {}, "ok")
            assert res == "ok"
            err = await mw.on_error(ValueError("x"), {})
            assert err is not None

        import asyncio
        asyncio.run(_run())


@pytest.mark.unit
class TestMiddlewarePipeline:
    @pytest.mark.asyncio
    async def test_pipeline_before_llm(self):
        req = _make_request()
        mw = ConcreteMiddleware()
        pipeline = MiddlewarePipeline([mw])
        out = await pipeline.run_before_llm(req)
        assert out is req

    @pytest.mark.asyncio
    async def test_pipeline_after_llm(self):
        req = _make_request()
        resp = _make_response()
        pipeline = MiddlewarePipeline([ConcreteMiddleware()])
        out = await pipeline.run_after_llm(req, resp)
        assert out is resp

    @pytest.mark.asyncio
    async def test_pipeline_before_tool(self):
        pipeline = MiddlewarePipeline([ConcreteMiddleware()])
        name, args = await pipeline.run_before_tool("echo", {"x": 1})
        assert name == "echo" and args == {"x": 1}

    @pytest.mark.asyncio
    async def test_pipeline_after_tool(self):
        pipeline = MiddlewarePipeline([ConcreteMiddleware()])
        result = await pipeline.run_after_tool("echo", {}, "done")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_pipeline_on_error(self):
        pipeline = MiddlewarePipeline([ConcreteMiddleware()])
        err = await pipeline.run_on_error(ValueError("x"), {"phase": "llm_call"})
        assert err is not None
        assert isinstance(err, ValueError)

    @pytest.mark.asyncio
    async def test_pipeline_ordering(self):
        order = []

        class M1(Middleware):
            async def before_llm_call(self, request):
                order.append("M1")
                return request

        class M2(Middleware):
            async def before_llm_call(self, request):
                order.append("M2")
                return request

        pipeline = MiddlewarePipeline([M1(), M2()])
        await pipeline.run_before_llm(_make_request())
        assert order == ["M1", "M2"]

    @pytest.mark.asyncio
    async def test_pipeline_error_suppression(self):
        class SuppressMiddleware(Middleware):
            async def on_error(self, error, context):
                return None

        pipeline = MiddlewarePipeline([SuppressMiddleware()])
        result = await pipeline.run_on_error(ValueError("x"), {})
        assert result is None

    @pytest.mark.asyncio
    async def test_pipeline_stream_chunk(self):
        from curio_agent_sdk.models.llm import LLMStreamChunk
        pipeline = MiddlewarePipeline([ConcreteMiddleware()])
        chunk = LLMStreamChunk(type="text_delta", text="hi")
        out = await pipeline.run_stream_chunk(_make_request(), chunk)
        assert out is chunk

    @pytest.mark.asyncio
    async def test_pipeline_stream_chunk_drop(self):
        from curio_agent_sdk.models.llm import LLMStreamChunk

        class DropChunkMiddleware(Middleware):
            async def on_llm_stream_chunk(self, request, chunk):
                return None

        pipeline = MiddlewarePipeline([DropChunkMiddleware()])
        chunk = LLMStreamChunk(type="text_delta", text="hi")
        out = await pipeline.run_stream_chunk(_make_request(), chunk)
        assert out is None

    @pytest.mark.asyncio
    async def test_pipeline_with_hook_registry_emits_before_after(self):
        registry = HookRegistry()
        seen = []

        def capture(ctx: HookContext):
            seen.append(ctx.event)

        registry.on(LLM_CALL_BEFORE, capture)
        registry.on(LLM_CALL_AFTER, capture)
        pipeline = MiddlewarePipeline([ConcreteMiddleware()], hook_registry=registry)
        req = _make_request()
        resp = _make_response()
        await pipeline.run_before_llm(req, run_id="r1", agent_id="a1")
        await pipeline.run_after_llm(req, resp, run_id="r1", agent_id="a1")
        assert LLM_CALL_BEFORE in seen and LLM_CALL_AFTER in seen

    @pytest.mark.asyncio
    async def test_pipeline_hook_cancel_raises(self):
        def cancel_llm(ctx: HookContext):
            ctx.cancel()

        registry = HookRegistry()
        registry.on(LLM_CALL_BEFORE, cancel_llm)
        pipeline = MiddlewarePipeline([ConcreteMiddleware()], hook_registry=registry)
        with pytest.raises(RuntimeError, match="cancelled"):
            await pipeline.run_before_llm(_make_request())

    @pytest.mark.asyncio
    async def test_wrap_llm_client(self):
        inner = AsyncMock()
        inner.call = AsyncMock(return_value=_make_response())
        pipeline = MiddlewarePipeline([ConcreteMiddleware()])
        wrapped = pipeline.wrap_llm_client(inner)
        req = _make_request()
        resp = await wrapped.call(req)
        assert resp is not None
        assert inner.call.called
        inner.call.assert_awaited_once()
