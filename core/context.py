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
        """
        Args:
            max_tokens: Maximum context window size (input + output).
            strategy: How to fit messages when over budget:
                - "truncate_oldest": Drop oldest messages (after system) until within budget.
                - "sliding_window": Same as truncate_oldest; keep most recent messages that fit.
                - "summarize": Replace truncated span with a single placeholder message
                  (optional summarizer callable can be added later for real summarization).
            reserve_tokens: Tokens to reserve for the model's response (default 1000).
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens
        self._budget = max(0, max_tokens - reserve_tokens)

    def count_tokens(self, messages: list[Message], model: str, tools: list | None = None) -> int:
        """
        Count tokens for a message list (and optional tool definitions) for the given model.

        Uses tiktoken for OpenAI/Groq, Anthropic's counting API for Anthropic,
        and approximate counting for other providers.
        """
        total = count_tokens_impl(messages, model, tools)
        return total

    def fit_messages(
        self,
        messages: list[Message],
        tools: list | None = None,
        model: str = "gpt-4o-mini",
    ) -> list[Message]:
        """
        Trim or summarize messages to fit within the token budget.

        Budget is max_tokens - reserve_tokens. When strategy is "truncate_oldest"
        or "sliding_window", oldest messages (after any system message) are dropped
        until the sequence fits. When strategy is "summarize", the dropped span is
        replaced by a single placeholder message (or a future summarizer callback).

        Args:
            messages: Full message history (system, user, assistant, tool).
            tools: Optional tool schemas (their token cost is included when counting).
            model: Model identifier for token counting (e.g. "openai:gpt-4o", "claude-sonnet-4-6").

        Returns:
            A new list of messages that fits within the budget.
        """
        if not messages:
            return []

        current = count_tokens_impl(messages, model, tools)
        if current <= self._budget:
            return list(messages)

        # Split system prefix from the rest (keep system at the start)
        system_messages: list[Message] = []
        rest: list[Message] = []
        for m in messages:
            if getattr(m, "role", None) == "system":
                system_messages.append(m)
            else:
                rest.append(m)

        if not rest:
            return list(messages)

        if self.strategy in ("truncate_oldest", "sliding_window"):
            return self._fit_truncate_oldest(system_messages, rest, tools, model)
        if self.strategy == "summarize":
            return self._fit_summarize(system_messages, rest, tools, model)
        return list(messages)

    def _fit_truncate_oldest(
        self,
        system_messages: list[Message],
        rest: list[Message],
        tools: list | None,
        model: str,
    ) -> list[Message]:
        """Drop oldest non-system messages until within budget."""
        out = list(system_messages)
        # Add from the end (most recent) until we would exceed budget
        for i in range(len(rest) - 1, -1, -1):
            candidate = [*system_messages, *rest[i:]]
            if count_tokens_impl(candidate, model, tools) <= self._budget:
                out = [*system_messages, *rest[i:]]
                break
        else:
            # Even a single latest message is over budget; keep it anyway
            out = [*system_messages, rest[-1]] if rest else out
        return out

    def _fit_summarize(
        self,
        system_messages: list[Message],
        rest: list[Message],
        tools: list | None,
        model: str,
    ) -> list[Message]:
        """Replace a truncated prefix with a single placeholder message."""
        out = list(system_messages)
        # Find how many trailing messages fit
        for i in range(len(rest) - 1, -1, -1):
            placeholder = Message.system(SUMMARIZE_PLACEHOLDER)
            candidate = [*system_messages, placeholder, *rest[i:]]
            if count_tokens_impl(candidate, model, tools) <= self._budget:
                out = [*system_messages, placeholder, *rest[i:]]
                break
        else:
            out = [*system_messages, rest[-1]] if rest else out
        return out
