"""
Unit tests for TracingMiddleware (no-op when OpenTelemetry not installed).
"""

import pytest

from curio_agent_sdk.middleware.tracing import TracingMiddleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, Message, TokenUsage


def _make_request():
    return LLMRequest(messages=[Message.user("hi")], model="gpt-4o", provider="openai")


def _make_response():
    return LLMResponse(
        message=Message.assistant("ok"),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="gpt-4o",
        provider="openai",
        finish_reason="stop",
    )


@pytest.mark.unit
class TestTracingMiddleware:
    @pytest.mark.asyncio
    async def test_span_creation_or_noop(self):
        mw = TracingMiddleware(service_name="test")
        req = _make_request()
        resp = _make_response()
        out_req = await mw.before_llm_call(req)
        assert out_req is req
        out_resp = await mw.after_llm_call(req, resp)
        assert out_resp is resp

    @pytest.mark.asyncio
    async def test_tool_call_passthrough(self):
        mw = TracingMiddleware(service_name="test")
        name, args = await mw.before_tool_call("echo", {"x": 1})
        assert name == "echo" and args.get("x") == 1
        result = await mw.after_tool_call("echo", {"x": 1}, "done")
        assert result == "done"
