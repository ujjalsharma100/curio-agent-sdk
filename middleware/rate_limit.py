"""
Client-side rate limiter middleware using a token bucket algorithm.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(Middleware):
    """
    Client-side token bucket rate limiter.

    Throttles LLM calls to stay within a configurable rate.
    Useful for avoiding provider rate limits proactively.

    The token bucket refills at `rate` tokens per second,
    up to `burst` tokens maximum. Each LLM call consumes 1 token.

    Example:
        # Allow 10 calls/second with burst of 20
        agent = Agent(
            middleware=[RateLimitMiddleware(rate=10.0, burst=20)],
            ...
        )
    """

    def __init__(self, rate: float = 10.0, burst: int = 10):
        """
        Args:
            rate: Tokens added per second (calls per second).
            burst: Maximum tokens in the bucket (max burst size).
        """
        self.rate = rate
        self.burst = burst
        self._tokens: float = float(burst)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        """Wait for a token before allowing the LLM call."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return request

            # Calculate wait time for 1 token
            wait_time = (1.0 - self._tokens) / self.rate
            logger.debug("Rate limit: waiting %.2fs for token", wait_time)

        # Wait outside the lock so other coroutines can proceed
        await asyncio.sleep(wait_time)

        async with self._lock:
            self._refill()
            self._tokens = max(0, self._tokens - 1.0)

        return request
