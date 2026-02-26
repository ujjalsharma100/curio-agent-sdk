"""
Middleware for the Curio Agent SDK.

Middleware intercepts LLM calls and tool calls, enabling cross-cutting
concerns like logging, cost tracking, rate limiting, tracing, and safety.

Example:
    from curio_agent_sdk.middleware import (
        LoggingMiddleware, CostTracker, TracingMiddleware, GuardrailsMiddleware,
    )

    agent = Agent(
        middleware=[
            LoggingMiddleware(),
            CostTracker(budget=1.0),
            TracingMiddleware(),
            GuardrailsMiddleware(block_patterns=[r"(?i)password"]),
        ],
        ...
    )
"""

from curio_agent_sdk.middleware.base import Middleware, MiddlewarePipeline
from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.middleware.cost_tracker import CostTracker
from curio_agent_sdk.middleware.rate_limit import RateLimitMiddleware
from curio_agent_sdk.middleware.tracing import TracingMiddleware
from curio_agent_sdk.middleware.guardrails import GuardrailsMiddleware, GuardrailsError

__all__ = [
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "CostTracker",
    "RateLimitMiddleware",
    "TracingMiddleware",
    "GuardrailsMiddleware",
    "GuardrailsError",
]
