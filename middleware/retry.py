"""
Retry middleware with configurable exponential backoff for LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse
from curio_agent_sdk.exceptions import LLMError, LLMRateLimitError

logger = logging.getLogger(__name__)


class RetryMiddleware(Middleware):
    """
    Retries failed LLM calls with exponential backoff.

    This middleware catches LLM errors and retries the call up to
    max_retries times with configurable backoff.

    Note: The LLMClient already has built-in failover across providers.
    This middleware adds retry logic *before* failover, useful for
    transient errors on the same provider.

    Example:
        agent = Agent(
            middleware=[RetryMiddleware(max_retries=3, base_delay=1.0)],
            ...
        )
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: tuple[type[Exception], ...] | None = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on or (LLMError,)
        self._attempt: int = 0

    async def on_error(self, error: Exception, context: dict[str, Any]) -> Exception | None:
        """
        Handle errors by tracking retry state.

        The actual retry is handled by the pipeline's call wrapper.
        This hook logs retry attempts.
        """
        if context.get("phase") != "llm_call":
            return error

        if not isinstance(error, self.retry_on):
            return error

        self._attempt += 1
        if self._attempt > self.max_retries:
            logger.error("Max retries (%d) exceeded for LLM call: %s", self.max_retries, error)
            self._attempt = 0
            return error

        delay = min(self.base_delay * (2 ** (self._attempt - 1)), self.max_delay)

        # Use retry_after from rate limit errors if available
        if isinstance(error, LLMRateLimitError) and error.retry_after:
            delay = max(delay, error.retry_after)

        logger.warning(
            "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
            self._attempt, self.max_retries, delay, error,
        )

        await asyncio.sleep(delay)

        # Return error to propagate (the LLMClient's own failover will handle it)
        return error

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        """Reset attempt counter on success."""
        self._attempt = 0
        return response
