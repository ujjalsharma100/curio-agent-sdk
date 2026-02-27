"""
Guardrails middleware for content safety filtering.

Provides regex-based filtering for both LLM inputs and outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.exceptions import CurioError
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class GuardrailsError(CurioError):
    """Raised when content is blocked by a guardrail rule."""

    def __init__(self, message: str, pattern: str, direction: str = "output"):
        self.pattern = pattern
        self.direction = direction
        super().__init__(message)


class GuardrailsMiddleware(Middleware):
    """
    Middleware that blocks LLM inputs or outputs matching regex patterns.

    Args:
        block_patterns: Regex patterns matched against LLM response content.
        block_input_patterns: Regex patterns matched against user message content.
        on_block: Optional callback invoked when content is blocked.

    Example:
        agent = Agent(
            middleware=[
                GuardrailsMiddleware(
                    block_patterns=[r"(?i)password", r"(?i)secret"],
                    block_input_patterns=[r"(?i)ignore previous instructions"],
                )
            ],
            ...
        )
    """

    def __init__(
        self,
        block_patterns: list[str] | None = None,
        block_input_patterns: list[str] | None = None,
        on_block: Callable[[GuardrailsError], Any] | None = None,
    ) -> None:
        self._block_patterns = [re.compile(p) for p in (block_patterns or [])]
        self._block_input_patterns = [re.compile(p) for p in (block_input_patterns or [])]
        self._on_block = on_block

    def _check_content(self, content: str, patterns: list[re.Pattern[str]], direction: str) -> None:
        for pattern in patterns:
            if pattern.search(content):
                error = GuardrailsError(
                    f"Content blocked by guardrail ({direction}): matched '{pattern.pattern}'",
                    pattern=pattern.pattern,
                    direction=direction,
                )
                logger.warning(str(error))
                if self._on_block is not None:
                    self._on_block(error)
                raise error

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        if not self._block_input_patterns:
            return request

        for message in reversed(request.messages):
            if message.role == "user":
                text = message.text
                if text:
                    self._check_content(text, self._block_input_patterns, "input")
                break
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        if not self._block_patterns:
            return response

        text = response.content
        if text:
            self._check_content(text, self._block_patterns, "output")
        return response
