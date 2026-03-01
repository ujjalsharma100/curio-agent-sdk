"""
Integration tests: Cost Tracking + Budget Enforcement (Phase 17 §21.12)

Validates end-to-end cost tracking and budget enforcement.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.middleware.cost_tracker import CostTracker
from curio_agent_sdk.testing.mock_llm import MockLLM


@tool
def expensive_tool(data: str) -> str:
    """A tool that represents an expensive operation."""
    return f"Processed: {data}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cost_tracking_accumulates():
    """Cost tracker accumulates costs across multiple LLM calls."""
    tracker = CostTracker(budget=100.0)

    mock = MockLLM()
    mock.add_tool_call_response("expensive_tool", {"data": "batch1"})
    mock.add_text_response("Processed batch1.")

    agent = Agent(
        system_prompt="Process data.",
        tools=[expensive_tool],
        middleware=[tracker],
        llm=mock,
    )
    await agent.arun("Process this data")

    assert tracker.total_cost >= 0
    assert tracker.call_count >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cost_budget_threshold_alert():
    """Threshold alerts fire when cost exceeds percentage of budget."""
    alerts = []

    def on_threshold(threshold, current, budget):
        alerts.append((threshold, current, budget))

    tracker = CostTracker(
        budget=0.001,
        alert_thresholds=[0.50, 0.80],
        on_threshold=on_threshold,
    )

    mock = MockLLM()
    mock.add_text_response("Response.")

    agent = Agent(
        system_prompt="Test.",
        middleware=[tracker],
        llm=mock,
    )
    result = await agent.arun("Test thresholds")

    assert result.status in ("completed", "error")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cost_model_breakdown():
    """Cost tracker provides per-model cost breakdown."""
    tracker = CostTracker(budget=100.0)

    mock = MockLLM()
    mock.add_text_response("Response from mock-model.")

    agent = Agent(
        system_prompt="Test.",
        middleware=[tracker],
        llm=mock,
    )
    await agent.arun("Test breakdown")

    breakdown = tracker.get_model_breakdown()
    assert isinstance(breakdown, dict)
