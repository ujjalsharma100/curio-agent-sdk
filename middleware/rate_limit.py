"""
Client-side rate limiter middleware using a token bucket algorithm.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class _TokenBucket:
    """Simple token bucket for a single scope (global, per-agent, per-user, etc.)."""

    tokens: float
    last_refill: float


class RateLimitMiddleware(Middleware):
    """
    Client-side token bucket rate limiter.

    Throttles LLM calls to stay within a configurable rate.
    Useful for avoiding provider rate limits proactively.

    The token bucket refills at `rate` tokens per second,
    up to `burst` tokens maximum. Each LLM call consumes 1 token.

    This implementation supports multi-tenant deployments by optionally
    maintaining separate buckets per agent, per user, and/or per routing
    tier using `LLMRequest.metadata`:

    - `agent_id` is expected in `request.metadata["agent_id"]`
    - `user_id` is expected in `request.metadata["user_id"]`
    - `tier` is taken from `request.tier`

    Example (global bucket, backwards compatible):
        agent = Agent(middleware=[RateLimitMiddleware(rate=10.0, burst=20)])

    Example (per-agent and per-user buckets):
        agent = Agent(middleware=[RateLimitMiddleware(
            rate=10.0,
            burst=20,
            per_agent=True,
            per_user=True,
        )])
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: int = 10,
        *,
        per_agent: bool = False,
        per_user: bool = False,
        per_tier: bool = False,
    ):
        """
        Args:
            rate: Tokens added per second (calls per second).
            burst: Maximum tokens in each bucket (max burst size).
            per_agent: Maintain a separate bucket per agent_id.
            per_user: Maintain a separate bucket per user_id.
            per_tier: Maintain a separate bucket per routing tier.
        """
        self.rate = rate
        self.burst = burst
        self.per_agent = per_agent
        self.per_user = per_user
        self.per_tier = per_tier
        # scope_key -> bucket
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = asyncio.Lock()

    def _bucket_key(self, request: LLMRequest) -> str:
        """
        Build a bucket key based on configured scopes and request metadata.

        Falls back to a single 'global' bucket when no scopes are enabled or
        metadata is missing.
        """
        parts: list[str] = ["global"]

        meta = request.metadata or {}
        if self.per_agent:
            agent_id = str(meta.get("agent_id") or "unknown-agent")
            parts.append(f"agent:{agent_id}")
        if self.per_user:
            user_id = str(meta.get("user_id") or "anonymous")
            parts.append(f"user:{user_id}")
        if self.per_tier:
            tier = str(request.tier or "default")
            parts.append(f"tier:{tier}")

        return "|".join(parts)

    def _refill(self, bucket: _TokenBucket) -> None:
        """Refill tokens in a bucket based on elapsed time."""
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
        bucket.last_refill = now

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        """Wait for a token before allowing the LLM call."""
        async with self._lock:
            key = self._bucket_key(request)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(tokens=float(self.burst), last_refill=time.monotonic())
                self._buckets[key] = bucket

            self._refill(bucket)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return request

            # Calculate wait time for 1 token
            wait_time = (1.0 - bucket.tokens) / self.rate
            logger.debug("Rate limit (%s): waiting %.2fs for token", key, wait_time)

        # Wait outside the lock so other coroutines can proceed
        await asyncio.sleep(wait_time)

        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(tokens=float(self.burst), last_refill=time.monotonic())
                self._buckets[key] = bucket
            self._refill(bucket)
            bucket.tokens = max(0.0, bucket.tokens - 1.0)

        return request
