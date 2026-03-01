"""
Unit tests for PrometheusExporter (no-op when prometheus_client not installed).
"""

import pytest

from curio_agent_sdk.middleware.prometheus import PrometheusExporter
from curio_agent_sdk.core.events import HookRegistry, HookContext
from curio_agent_sdk.core.events.hooks import LLM_CALL_BEFORE, LLM_CALL_AFTER


@pytest.mark.unit
class TestPrometheusExporter:
    def test_metrics_recorded_or_noop(self):
        exporter = PrometheusExporter(namespace="curio_test")
        registry = HookRegistry()
        exporter.attach(registry)
        exporter.detach(registry)

    @pytest.mark.asyncio
    async def test_counter_incremented_or_noop(self):
        exporter = PrometheusExporter(namespace="curio_test")
        registry = HookRegistry()
        exporter.attach(registry)
        ctx = HookContext(
            event=LLM_CALL_BEFORE,
            data={"request": None},
            run_id="r1",
        )
        await registry.emit(LLM_CALL_BEFORE, ctx)
        ctx.data["response"] = type("R", (), {"provider": "openai", "model": "gpt-4o", "usage": type("U", (), {"input_tokens": 10, "output_tokens": 5})()})()
        await registry.emit(LLM_CALL_AFTER, ctx)
        exporter.detach(registry)
