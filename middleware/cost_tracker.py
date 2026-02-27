"""
Cost tracking middleware with budget enforcement.

Tracks cumulative cost of LLM calls and raises CostBudgetExceeded
when a configured budget is reached.
"""

from __future__ import annotations

import logging
from typing import Any

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse
from curio_agent_sdk.models.exceptions import CostBudgetExceeded

logger = logging.getLogger(__name__)

# Approximate cost per 1M tokens (USD) for common models.
# Used as defaults; users can override with custom pricing.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # model_pattern: {"input": cost_per_1M, "output": cost_per_1M}
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-4": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
}


class CostTracker(Middleware):
    """
    Tracks LLM call costs and enforces a budget limit.

    Raises CostBudgetExceeded before an LLM call if the accumulated
    cost has already exceeded the budget.

    Example:
        tracker = CostTracker(budget=1.00)  # $1.00 budget
        agent = Agent(middleware=[tracker], ...)
        result = await agent.arun("...")
        print(f"Total cost: ${tracker.total_cost:.4f}")
    """

    def __init__(
        self,
        budget: float | None = None,
        pricing: dict[str, dict[str, float]] | None = None,
    ):
        self.budget = budget
        self.pricing = {**DEFAULT_PRICING, **(pricing or {})}
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Look up pricing for a model, with fuzzy matching."""
        if model in self.pricing:
            return self.pricing[model]
        # Try matching by substring
        for pattern, price in self.pricing.items():
            if pattern in model or model in pattern:
                return price
        # Default: approximate mid-range pricing
        return {"input": 1.00, "output": 3.00}

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a single LLM call."""
        pricing = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        """Check budget before making an LLM call."""
        if self.budget is not None and self.total_cost >= self.budget:
            raise CostBudgetExceeded(self.total_cost, self.budget)
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        """Track cost after an LLM call."""
        cost = self._calculate_cost(
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        self.total_cost += cost
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.call_count += 1

        logger.debug(
            "LLM cost: $%.4f (total: $%.4f / $%.4f budget) | model=%s",
            cost, self.total_cost, self.budget or float('inf'), response.model,
        )

        # Check budget after call too (for next iteration)
        if self.budget is not None and self.total_cost >= self.budget:
            logger.warning("Cost budget exceeded: $%.4f >= $%.4f", self.total_cost, self.budget)

        return response

    def reset(self):
        """Reset the cost tracker."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
