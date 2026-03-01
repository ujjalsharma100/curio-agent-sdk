"""
Unit tests for LoggingMiddleware.
"""

import logging
import pytest

from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, Message, TokenUsage


def _make_request():
    return LLMRequest(messages=[Message.user("hello")], model="gpt-4o", provider="openai")


def _make_response():
    return LLMResponse(
        message=Message.assistant("hi"),
        usage=TokenUsage(input_tokens=5, output_tokens=3),
        model="gpt-4o",
        provider="openai",
        finish_reason="stop",
    )


@pytest.mark.unit
class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_logs_llm_call(self, caplog):
        caplog.set_level(logging.INFO)
        mw = LoggingMiddleware(level=logging.INFO)
        req = _make_request()
        resp = _make_response()
        await mw.before_llm_call(req)
        await mw.after_llm_call(req, resp)
        assert "LLM call started" in caplog.text or "model=" in caplog.text
        assert "LLM call completed" in caplog.text or "finish=" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_tool_call(self, caplog):
        caplog.set_level(logging.INFO)
        mw = LoggingMiddleware(level=logging.INFO)
        await mw.before_tool_call("calculator", {"x": 1})
        await mw.after_tool_call("calculator", {"x": 1}, "result")
        assert "Tool call" in caplog.text or "tool=" in caplog.text

    @pytest.mark.asyncio
    async def test_log_format(self):
        mw = LoggingMiddleware(level=logging.DEBUG, logger_name="test.logger")
        assert mw.level == logging.DEBUG
        assert mw.log.name == "test.logger"
        req = _make_request()
        out = await mw.before_llm_call(req)
        assert out is req
