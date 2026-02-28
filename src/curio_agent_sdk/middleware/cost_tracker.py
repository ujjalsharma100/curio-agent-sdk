"""
Cost tracking middleware with budget enforcement.

Tracks cumulative cost of LLM calls and raises CostBudgetExceeded
when a configured budget is reached. Supports per-model breakdown,
threshold alerts, and optional persistence of cost entries.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse
from curio_agent_sdk.models.exceptions import CostBudgetExceeded

if TYPE_CHECKING:
    from curio_agent_sdk.persistence.base import BasePersistence

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

    Args:
        budget: Maximum allowed cost in USD. None = unlimited.
        pricing: Custom pricing overrides (model -> {input, output} per 1M tokens).
        persistence: Optional persistence backend to write cost entries.
        alert_thresholds: Budget fraction thresholds that trigger alerts
            (e.g. ``[0.50, 0.80, 0.95]`` = warn at 50%, 80%, 95%).
        on_threshold: Callback invoked when a threshold is crossed.
            Signature: ``callback(threshold, current_cost, budget)``.

    Example::

        tracker = CostTracker(
            budget=1.00,
            alert_thresholds=[0.50, 0.80, 0.95],
            on_threshold=lambda t, c, b: print(f"Alert: {t*100:.0f}% budget used"),
        )
        agent = Agent(middleware=[tracker], ...)
        result = await agent.arun("...")
        print(f"Total cost: ${tracker.total_cost:.4f}")
        print(tracker.get_model_breakdown())
    """

    def __init__(
        self,
        budget: float | None = None,
        pricing: dict[str, dict[str, float]] | None = None,
        persistence: "BasePersistence | None" = None,
        alert_thresholds: list[float] | None = None,
        on_threshold: Callable[[float, float, float], None] | None = None,
    ):
        self.budget = budget
        self.pricing = {**DEFAULT_PRICING, **(pricing or {})}
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0

        # Per-model tracking
        self.per_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        )

        # Persistence
        self._persistence = persistence

        # Threshold alerts
        self._alert_thresholds = sorted(alert_thresholds or [])
        self._on_threshold = on_threshold
        self._crossed_thresholds: set[float] = set()

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

    def _check_thresholds(self) -> None:
        """Fire alerts for any newly-crossed budget thresholds."""
        if self.budget is None or self.budget <= 0:
            return
        fraction = self.total_cost / self.budget
        for threshold in self._alert_thresholds:
            if threshold in self._crossed_thresholds:
                continue
            if fraction >= threshold:
                self._crossed_thresholds.add(threshold)
                logger.warning(
                    "Cost threshold crossed: %.0f%% of $%.2f budget (current: $%.4f)",
                    threshold * 100,
                    self.budget,
                    self.total_cost,
                )
                if self._on_threshold:
                    try:
                        self._on_threshold(threshold, self.total_cost, self.budget)
                    except Exception as e:
                        logger.error("Threshold callback error: %s", e)

    def _persist_cost_entry(self, model: str, cost: float, input_tokens: int, output_tokens: int) -> None:
        """Write a cost entry to persistence if configured."""
        if self._persistence is None:
            return
        try:
            from curio_agent_sdk.models.agent import AgentRunEvent
            self._persistence.log_agent_run_event(AgentRunEvent(
                agent_id="",
                run_id="",
                agent_name="",
                timestamp=datetime.now(),
                event_type="cost_entry",
                data=json.dumps({
                    "model": model,
                    "cost_usd": round(cost, 6),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cumulative_cost_usd": round(self.total_cost, 6),
                }),
            ))
        except Exception as e:
            logger.warning("Failed to persist cost entry: %s", e)

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        """Check budget before making an LLM call."""
        if self.budget is not None and self.total_cost >= self.budget:
            raise CostBudgetExceeded(self.total_cost, self.budget)
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        """Track cost after an LLM call."""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        model = response.model

        cost = self._calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        # Per-model tracking
        entry = self.per_model[model]
        entry["cost"] += cost
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["calls"] += 1

        logger.debug(
            "LLM cost: $%.4f (total: $%.4f / $%.4f budget) | model=%s",
            cost, self.total_cost, self.budget or float('inf'), model,
        )

        # Check budget after call too (for next iteration)
        if self.budget is not None and self.total_cost >= self.budget:
            logger.warning("Cost budget exceeded: $%.4f >= $%.4f", self.total_cost, self.budget)

        # Threshold alerts
        self._check_thresholds()

        # Persist cost entry
        self._persist_cost_entry(model, cost, input_tokens, output_tokens)

        return response

    def get_model_breakdown(self) -> dict[str, dict[str, Any]]:
        """
        Return per-model cost/token/call breakdown.

        Returns:
            Dict mapping model name to ``{cost, input_tokens, output_tokens, calls}``.
        """
        return dict(self.per_model)

    def get_summary(self) -> dict[str, Any]:
        """
        Return an overall cost summary.

        Returns:
            Dict with ``total_cost``, ``total_input_tokens``, ``total_output_tokens``,
            ``call_count``, ``budget``, ``budget_remaining``, ``per_model``.
        """
        return {
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "budget": self.budget,
            "budget_remaining": round(self.budget - self.total_cost, 6) if self.budget else None,
            "per_model": self.get_model_breakdown(),
        }

    def reset(self):
        """Reset the cost tracker."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.per_model.clear()
        self._crossed_thresholds.clear()
