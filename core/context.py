"""
Context window management: fit message history within token budgets.

Provides ContextManager for trimming or summarizing messages so that
conversation + tools fit within a model's context limit.
"""

from __future__ import annotations

import logging
from typing import Literal

from curio_agent_sdk.llm.token_counter import count_tokens as count_tokens_impl
from curio_agent_sdk.models.llm import Message

logger = logging.getLogger(__name__)

# Placeholder text when using "summarize" strategy without a summarizer callback
SUMMARIZE_PLACEHOLDER = "[Earlier messages were truncated due to context length.]"


class ContextManager:
    """
    Manages message history within token budgets.

    Trims or summarizes messages so that the conversation fits within
    max_tokens, reserving reserve_tokens for the model's response.
    """

    def __init__(
        self,
        max_tokens: int,
        strategy: Literal["truncate_oldest", "summarize", "sliding_window"] = "truncate_oldest",
        reserve_tokens: int = 1000,
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens
        self._budget = max(0, max_tokens - reserve_tokens)

    def count_tokens(self, messages: list[Message], model: str, tools: list | None = None) -> int:
        """Count tokens for a message list (and optional tool definitions) for the given model."""
        return count_tokens_impl(messages, model, tools)

    def fit_messages(
        self,
        messages: list[Message],
        tools: list | None = None,
        model: str = "gpt-4o-mini",
    ) -> list[Message]:
        """
        Trim or summarize messages to fit within the token budget.

        Budget is max_tokens - reserve_tokens. Only the first system message
        is preserved at the start; all others are part of the trimmable history.

        Args:
            messages: Full message history (system, user, assistant, tool).
            tools: Optional tool schemas (their token cost is included when counting).
            model: Model identifier for token counting.

        Returns:
            A new list of messages that fits within the budget.
        """
        if not messages:
            return []

        current = count_tokens_impl(messages, model, tools)
        if current <= self._budget:
            return list(messages)

        # Only preserve the first system message (standard convention)
        system_msg: Message | None = None
        rest: list[Message] = []
        for m in messages:
            if system_msg is None and getattr(m, "role", None) == "system":
                system_msg = m
            else:
                rest.append(m)

        if not rest:
            return list(messages)

        system_prefix = [system_msg] if system_msg else []

        if self.strategy in ("truncate_oldest", "sliding_window"):
            return self._fit_truncate_oldest(system_prefix, rest, tools, model)
        if self.strategy == "summarize":
            return self._fit_summarize(system_prefix, rest, tools, model)
        return list(messages)

    def _fit_truncate_oldest(
        self,
        system_prefix: list[Message],
        rest: list[Message],
        tools: list | None,
        model: str,
    ) -> list[Message]:
        """Drop oldest non-system messages until within budget using binary search."""
        n = len(rest)

        # Count system prefix tokens once
        system_tokens = count_tokens_impl(system_prefix, model, tools) if system_prefix else 0

        # Binary search for the earliest start index where messages fit
        # We want the smallest i such that system_prefix + rest[i:] fits in budget
        lo, hi = 0, n - 1
        best_start = n - 1  # fallback: keep only the last message

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = [*system_prefix, *rest[mid:]]
            token_count = count_tokens_impl(candidate, model, tools)
            if token_count <= self._budget:
                best_start = mid
                hi = mid - 1  # try to include more messages
            else:
                lo = mid + 1  # need to drop more

        return [*system_prefix, *rest[best_start:]]

    def _fit_summarize(
        self,
        system_prefix: list[Message],
        rest: list[Message],
        tools: list | None,
        model: str,
    ) -> list[Message]:
        """Replace a truncated prefix with a single placeholder message."""
        n = len(rest)
        placeholder = Message.system(SUMMARIZE_PLACEHOLDER)

        # Binary search for earliest start index with placeholder
        lo, hi = 0, n - 1
        best_start = n - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = [*system_prefix, placeholder, *rest[mid:]]
            token_count = count_tokens_impl(candidate, model, tools)
            if token_count <= self._budget:
                best_start = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if best_start == 0:
            # Everything fits with placeholder, but we already checked without it
            # so just return with placeholder
            return [*system_prefix, placeholder, *rest]

        return [*system_prefix, placeholder, *rest[best_start:]]
