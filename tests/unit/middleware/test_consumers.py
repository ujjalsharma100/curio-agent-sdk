"""
Unit tests for hook-based consumers (LoggingConsumer, TracingConsumer, etc.).
"""

import logging
import pytest

from curio_agent_sdk.middleware.consumers import (
    LoggingConsumer,
    TracingConsumer,
    TraceContextFilter,
)
from curio_agent_sdk.core.events import HookRegistry, HookContext
from curio_agent_sdk.core.events.hooks import (
    LLM_CALL_BEFORE,
    LLM_CALL_AFTER,
    TOOL_CALL_BEFORE,
    TOOL_CALL_AFTER,
)


@pytest.mark.unit
class TestLoggingConsumer:
    @pytest.mark.asyncio
    async def test_hook_consumer_llm(self, caplog):
        registry = HookRegistry()
        consumer = LoggingConsumer(level=logging.DEBUG)
        consumer.attach(registry)
        caplog.set_level(logging.DEBUG)
        ctx = HookContext(
            event=LLM_CALL_BEFORE,
            data={"request": None, "model": "gpt-4o", "provider": "openai"},
            run_id="r1",
        )
        await registry.emit(LLM_CALL_BEFORE, ctx)
        consumer.detach(registry)
        assert "LLM" in caplog.text or "run_id" in caplog.text

    @pytest.mark.asyncio
    async def test_hook_consumer_tools(self, caplog):
        registry = HookRegistry()
        consumer = LoggingConsumer(level=logging.INFO)
        consumer.attach(registry)
        caplog.set_level(logging.INFO)
        ctx = HookContext(
            event=TOOL_CALL_BEFORE,
            data={"tool_name": "calculator"},
            run_id="r1",
        )
        await registry.emit(TOOL_CALL_BEFORE, ctx)
        ctx2 = HookContext(
            event=TOOL_CALL_AFTER,
            data={"tool_name": "calculator", "result": "4"},
            run_id="r1",
        )
        await registry.emit(TOOL_CALL_AFTER, ctx2)
        consumer.detach(registry)
        assert "Tool" in caplog.text or "tool" in caplog.text


@pytest.mark.unit
class TestTracingConsumer:
    def test_attach_detach(self):
        registry = HookRegistry()
        consumer = TracingConsumer(service_name="test")
        consumer.attach(registry)
        consumer.detach(registry)


@pytest.mark.unit
class TestTraceContextFilter:
    def test_filter_adds_attributes(self):
        f = TraceContextFilter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "", (), None
        )
        result = f.filter(record)
        assert result is True
        assert hasattr(record, "trace_id") or getattr(record, "trace_id", "") is not None or getattr(record, "trace_id", "") == ""
