"""
Unit tests for CostTracker middleware.
"""

import pytest

from curio_agent_sdk.middleware.cost_tracker import CostTracker, DEFAULT_PRICING
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, Message, TokenUsage
from curio_agent_sdk.models.exceptions import CostBudgetExceeded


def _make_request(model="gpt-4o"):
    return LLMRequest(messages=[Message.user("hi")], model=model, provider="openai")


def _make_response(model="gpt-4o", input_tokens=1000, output_tokens=500):
    return LLMResponse(
        message=Message.assistant("ok"),
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        provider="openai",
        finish_reason="stop",
    )


@pytest.mark.unit
class TestCostTracker:
    @pytest.mark.asyncio
    async def test_cost_tracking_openai(self):
        tracker = CostTracker()
        req = _make_request("gpt-4o")
        resp = _make_response("gpt-4o", 1000, 500)
        await tracker.before_llm_call(req)
        await tracker.after_llm_call(req, resp)
        assert tracker.total_cost > 0
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert "gpt-4o" in tracker.per_model

    @pytest.mark.asyncio
    async def test_cost_tracking_anthropic(self):
        tracker = CostTracker()
        req = _make_request("claude-sonnet-4")
        resp = _make_response("claude-sonnet-4", 500, 200)
        await tracker.before_llm_call(req)
        await tracker.after_llm_call(req, resp)
        assert tracker.total_cost > 0

    @pytest.mark.asyncio
    async def test_cost_budget_enforcement(self):
        tracker = CostTracker(budget=0.001)
        req = _make_request("gpt-4o")
        resp = _make_response("gpt-4o", 100_000, 50_000)
        await tracker.before_llm_call(req)
        await tracker.after_llm_call(req, resp)
        with pytest.raises(CostBudgetExceeded):
            await tracker.before_llm_call(req)

    @pytest.mark.asyncio
    async def test_cost_tracking_accumulation(self):
        tracker = CostTracker()
        for _ in range(3):
            req = _make_request()
            resp = _make_response("gpt-4o", 100, 50)
            await tracker.before_llm_call(req)
            await tracker.after_llm_call(req, resp)
        assert tracker.call_count == 3
        assert tracker.total_cost > 0

    @pytest.mark.asyncio
    async def test_cost_unknown_model(self):
        tracker = CostTracker()
        req = _make_request("unknown-model-xyz")
        resp = _make_response("unknown-model-xyz", 100, 50)
        await tracker.before_llm_call(req)
        await tracker.after_llm_call(req, resp)
        assert tracker.total_cost > 0

    def test_cost_reset(self):
        tracker = CostTracker()
        tracker.total_cost = 1.0
        tracker.total_input_tokens = 100
        tracker.call_count = 5
        tracker.reset()
        assert tracker.total_cost == 0.0
        assert tracker.total_input_tokens == 0
        assert tracker.call_count == 0
        assert not tracker.per_model

    def test_get_model_breakdown(self):
        tracker = CostTracker()
        tracker.per_model["gpt-4o"] = {"cost": 0.01, "input_tokens": 100, "output_tokens": 50, "calls": 1}
        breakdown = tracker.get_model_breakdown()
        assert "gpt-4o" in breakdown
        assert breakdown["gpt-4o"]["cost"] == 0.01

    def test_get_summary(self):
        tracker = CostTracker(budget=1.0)
        tracker.total_cost = 0.5
        summary = tracker.get_summary()
        assert summary["total_cost"] == 0.5
        assert summary["budget"] == 1.0
        assert summary["budget_remaining"] == 0.5
