"""
Middleware for the Curio Agent SDK.

Middleware intercepts LLM calls and tool calls, enabling cross-cutting
concerns like logging, cost tracking, rate limiting, and retry logic.

Example:
    from curio_agent_sdk.middleware import LoggingMiddleware, CostTracker

    agent = Agent(
        middleware=[LoggingMiddleware(), CostTracker(budget=1.0)],
        ...
    )
"""

from curio_agent_sdk.middleware.base import Middleware, MiddlewarePipeline
from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.middleware.cost_tracker import CostTracker
from curio_agent_sdk.middleware.retry import RetryMiddleware
from curio_agent_sdk.middleware.rate_limit import RateLimitMiddleware

__all__ = [
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "CostTracker",
    "RetryMiddleware",
    "RateLimitMiddleware",
]
