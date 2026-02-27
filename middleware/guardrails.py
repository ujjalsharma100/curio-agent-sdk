"""
Guardrails middleware for content safety filtering.

Provides regex-based filtering for both LLM inputs and outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Tuple

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.exceptions import CurioError
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, Message

logger = logging.getLogger(__name__)


class GuardrailsError(CurioError):
    """Raised when content is blocked by a guardrail rule."""

    def __init__(self, message: str, pattern: str, direction: str = "output"):
        self.pattern = pattern
        self.direction = direction
        super().__init__(message)


class GuardrailsMiddleware(Middleware):
    """
    Middleware that blocks LLM inputs or outputs matching regex patterns and
    optionally performs heuristic prompt-injection detection.

    Args:
        block_patterns: Regex patterns matched against LLM response content.
        block_input_patterns: Regex patterns matched against user message content.
        on_block: Optional callback invoked when content is blocked.
        block_prompt_injection: When True, apply heuristic detection to user
            prompts and block when they look like prompt injection attempts.
    """

    def __init__(
        self,
        block_patterns: list[str] | None = None,
        block_input_patterns: list[str] | None = None,
        on_block: Callable[[GuardrailsError], Any] | None = None,
        block_prompt_injection: bool = False,
    ) -> None:
        self._block_patterns = [re.compile(p) for p in (block_patterns or [])]
        self._block_input_patterns = [re.compile(p) for p in (block_input_patterns or [])]
        self._on_block = on_block
        self._block_prompt_injection = block_prompt_injection

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

    def _looks_like_prompt_injection(self, text: str) -> bool:
        """
        Heuristic prompt injection detection.

        This is intentionally conservative and pattern-based; for more advanced
        detection callers can layer an LLM-based classifier via hooks.
        """
        lowered = text.lower()
        suspicious_phrases = [
            "ignore previous instructions",
            "forget previous instructions",
            "disregard all prior rules",
            "you are no longer",
            "you must now act as",
            "as a large language model you must ignore",
            "developer mode",
            "system prompt",
        ]
        return any(phrase in lowered for phrase in suspicious_phrases)

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        # Start from the last user message in the request
        last_user: Message | None = None
        for message in reversed(request.messages):
            if message.role == "user":
                last_user = message
                break

        if last_user is None:
            return request

        text = last_user.text
        if not text:
            return request

        # Regex-based input blocking
        if self._block_input_patterns:
            self._check_content(text, self._block_input_patterns, "input")

        # Heuristic prompt injection detection
        if self._block_prompt_injection and self._looks_like_prompt_injection(text):
            error = GuardrailsError(
                "Content blocked by guardrail (input): suspected prompt injection",
                pattern="prompt_injection_heuristic",
                direction="input",
            )
            logger.warning(str(error))
            if self._on_block is not None:
                self._on_block(error)
            raise error

        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        if not self._block_patterns:
            return response

        text = response.content
        if text:
            self._check_content(text, self._block_patterns, "output")
        return response


class PIIMiddleware(Middleware):
    """
    Middleware that detects and redacts common PII patterns in inputs and outputs.

    Redaction is applied in-place so that downstream components only see
    sanitized content.
    """

    def __init__(
        self,
        *,
        redact_input: bool = True,
        redact_output: bool = True,
        replacement: str = "[REDACTED]",
        extra_patterns: list[str] | None = None,
    ) -> None:
        self.redact_input = redact_input
        self.redact_output = redact_output
        self.replacement = replacement

        patterns: list[str] = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
            r"\b\d{16}\b",  # Simple 16-digit credit card number
            r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b",  # Grouped credit card
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # Email
            r"\b\+?\d{1,3}[-.\s]??\d{3}[-.\s]??\d{3,4}[-.\s]??\d{3,4}\b",  # Phone-ish
        ]
        if extra_patterns:
            patterns.extend(extra_patterns)
        self._patterns = [re.compile(p) for p in patterns]

    def _redact(self, text: str) -> Tuple[str, bool]:
        """Redact PII patterns from text. Returns (redacted_text, changed_flag)."""
        original = text
        for pattern in self._patterns:
            text = pattern.sub(self.replacement, text)
        return text, text != original

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        if not self.redact_input:
            return request

        for i in range(len(request.messages) - 1, -1, -1):
            msg = request.messages[i]
            if msg.role == "user":
                if isinstance(msg.content, str):
                    new_text, changed = self._redact(msg.content)
                    if changed:
                        request.messages[i] = Message.user(new_text)
                break
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        if not self.redact_output:
            return response

        text = response.content
        if not text:
            return response

        new_text, changed = self._redact(text)
        if changed:
            response.message = Message.assistant(
                new_text,
                tool_calls=response.message.tool_calls,
            )
        return response


class ContentPolicyMiddleware(Middleware):
    """
    Middleware that enforces simple content policies on LLM outputs.

    Policies are expressed as categories mapped to regex patterns. When a
    category is blocked and a pattern matches the response content, the
    middleware raises `GuardrailsError`.
    """

    def __init__(
        self,
        blocked_categories: list[str] | None = None,
        category_patterns: Dict[str, list[str]] | None = None,
        on_block: Callable[[GuardrailsError], Any] | None = None,
    ) -> None:
        self.blocked_categories = set(blocked_categories or [])
        default_patterns: Dict[str, list[str]] = {
            "self_harm": [
                r"suicide",
                r"kill myself",
                r"end my life",
            ],
            "violence": [
                r"kill (?:him|her|them|someone)",
                r"make a bomb",
            ],
            "hate": [
                r"\bhate speech\b",
                r"\bslur\b",
            ],
        }
        if category_patterns:
            for k, v in category_patterns.items():
                default_patterns.setdefault(k, []).extend(v)

        self._compiled: Dict[str, list[re.Pattern[str]]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in pats]
            for cat, pats in default_patterns.items()
        }
        self._on_block = on_block

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        if not self.blocked_categories:
            return response

        text = response.content
        if not text:
            return response

        lowered = text.lower()
        for category in self.blocked_categories:
            patterns = self._compiled.get(category, [])
            for pattern in patterns:
                if pattern.search(lowered):
                    error = GuardrailsError(
                        f"Content blocked by policy: category '{category}'",
                        pattern=pattern.pattern,
                        direction="output",
                    )
                    logger.warning(str(error))
                    if self._on_block is not None:
                        self._on_block(error)
                    raise error
        return response
