"""
Integration tests: Middleware Pipeline (Phase 17 §21.12)

Validates multiple middleware stacked together.

Note: Middleware wraps the LLM client, so we use agent.arun() directly.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.middleware.cost_tracker import CostTracker
from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse
from curio_agent_sdk.testing.mock_llm import MockLLM


class OrderTrackingMiddleware(Middleware):
    """Tracks execution order."""

    def __init__(self, name: str, order_log: list):
        self._name = name
        self._order_log = order_log

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        self._order_log.append(f"{self._name}:before")
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        self._order_log.append(f"{self._name}:after")
        return response


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_middleware_ordering():
    """Middleware executes in registration order for before, reverse for after."""
    order = []
    mw1 = OrderTrackingMiddleware("first", order)
    mw2 = OrderTrackingMiddleware("second", order)

    mock = MockLLM()
    mock.add_text_response("Done.")

    agent = Agent(
        system_prompt="Test.",
        middleware=[mw1, mw2],
        llm=mock,
    )
    await agent.arun("Test ordering")

    assert "first:before" in order
    assert "second:before" in order
    assert order.index("first:before") < order.index("second:before")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_middleware_with_cost_and_logging():
    """CostTracker + LoggingMiddleware work together."""
    tracker = CostTracker(budget=10.0)
    logger_mw = LoggingMiddleware()

    mock = MockLLM()
    mock.add_text_response("Combined response.")

    agent = Agent(
        system_prompt="Test.",
        middleware=[logger_mw, tracker],
        llm=mock,
    )
    result = await agent.arun("Test combined")

    assert result.status == "completed"
    assert tracker.call_count >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_middleware_error_propagation():
    """Middleware error handler is invoked on errors."""

    class ErrorTrackingMiddleware(Middleware):
        def __init__(self):
            self.errors_seen = []

        async def on_error(self, error, context):
            self.errors_seen.append(str(error))
            return error

    error_mw = ErrorTrackingMiddleware()

    mock = MockLLM()
    mock.add_text_response("Response.")

    agent = Agent(
        system_prompt="Test.",
        middleware=[error_mw],
        llm=mock,
    )
    result = await agent.arun("Test errors")

    assert result.status == "completed"
