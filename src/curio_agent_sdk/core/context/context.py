"""
Context window management: fit message history within token budgets.

Provides ContextManager for trimming or summarizing messages so that
conversation + tools fit within a model's context limit.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

from curio_agent_sdk.core.llm.token_counter import count_tokens as count_tokens_impl
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
        strategy: Literal["truncate_oldest", "summarize"] = "truncate_oldest",
        reserve_tokens: int = 1000,
        summarizer: Callable[[list[Message], str, list | None], Message] | None = None,
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens
        self._budget = max(0, max_tokens - reserve_tokens)
        # Optional summarizer callback used when strategy="summarize".
        # The callback receives (messages_to_summarize, model, tools) and
        # should return a single Message (typically system or assistant).
        self.summarizer = summarizer

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

        if self.strategy == "truncate_oldest":
            return self._fit_truncate_oldest(system_prefix, rest, tools, model)
        if self.strategy == "summarize":
            return self._fit_summarize(system_prefix, rest, tools, model)
        return list(messages)

    def _group_messages(self, rest: list[Message]) -> list[list[Message]]:
        """
        Group messages into atomic windows that should not be split.

        In particular, assistant messages that initiate tool calls are grouped
        together with their corresponding tool-result messages so that tool
        call / result pairs are preserved when trimming.
        """
        groups: list[list[Message]] = []
        i = 0
        n = len(rest)

        while i < n:
            msg = rest[i]
            role = getattr(msg, "role", None)

            # Group assistant tool calls with their subsequent tool results
            if role == "assistant" and getattr(msg, "tool_calls", None):
                group = [msg]
                tool_ids = {
                    getattr(tc, "id", "")
                    for tc in (msg.tool_calls or [])
                    if getattr(tc, "id", None)
                }
                j = i + 1
                while j < n:
                    next_msg = rest[j]
                    if getattr(next_msg, "role", None) != "tool":
                        break
                    tc_id = getattr(next_msg, "tool_call_id", None)
                    if not tc_id or tc_id not in tool_ids:
                        break
                    group.append(next_msg)
                    j += 1
                groups.append(group)
                i = j
                continue

            # Default: single-message group
            groups.append([msg])
            i += 1

        return groups

    def _fit_truncate_oldest(
        self,
        system_prefix: list[Message],
        rest: list[Message],
        tools: list | None,
        model: str,
    ) -> list[Message]:
        """
        Drop oldest non-system messages until within budget using binary search.

        Uses a token-aware sliding window over grouped messages so that
        tool call / result pairs are preserved.
        """
        groups = self._group_messages(rest)
        if not groups:
            return list(system_prefix)

        n = len(groups)

        # Binary search for the earliest group index where messages fit.
        # We want the smallest g such that system_prefix + groups[g:] fits.
        lo, hi = 0, n - 1
        best_start = n - 1  # fallback: keep only the last group

        while lo <= hi:
            mid = (lo + hi) // 2
            tail_messages = [m for group in groups[mid:] for m in group]
            candidate = [*system_prefix, *tail_messages]
            token_count = count_tokens_impl(candidate, model, tools)
            if token_count <= self._budget:
                best_start = mid
                hi = mid - 1  # try to include more groups
            else:
                lo = mid + 1  # need to drop more

        tail_messages = [m for group in groups[best_start:] for m in group]
        return [*system_prefix, *tail_messages]

    def _fit_summarize(
        self,
        system_prefix: list[Message],
        rest: list[Message],
        tools: list | None,
        model: str,
    ) -> list[Message]:
        """
        Replace a truncated prefix with a single summary message.

        If a summarizer callback is provided, it is used to generate an
        LLM-based summary of the truncated prefix; otherwise a placeholder
        system message is used.
        """
        groups = self._group_messages(rest)
        if not groups:
            return list(system_prefix)

        n = len(groups)
        placeholder = Message.system(SUMMARIZE_PLACEHOLDER)

        # Binary search for earliest group index where a placeholder summary +
        # remaining groups fits within the budget.
        lo, hi = 0, n - 1
        best_start = n - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            tail_messages = [m for group in groups[mid:] for m in group]
            candidate = [*system_prefix, placeholder, *tail_messages]
            token_count = count_tokens_impl(candidate, model, tools)
            if token_count <= self._budget:
                best_start = mid
                hi = mid - 1
            else:
                lo = mid + 1

        # Messages to summarize (prefix) and messages to keep as-is (tail)
        prefix_groups = groups[:best_start]
        tail_groups = groups[best_start:]

        if not prefix_groups:
            # Nothing to summarize; just fall back to truncate_oldest behavior
            flat_rest = [m for group in groups for m in group]
            return self._fit_truncate_oldest(system_prefix, flat_rest, tools, model)

        prefix_messages = [m for group in prefix_groups for m in group]
        tail_messages = [m for group in tail_groups for m in group]

        # Generate an actual summary message if a summarizer is provided.
        if self.summarizer is not None:
            try:
                summary_msg = self.summarizer(prefix_messages, model, tools)
            except Exception as e:
                logger.warning("Context summarizer failed (%s); using placeholder", e)
                summary_msg = Message.system(SUMMARIZE_PLACEHOLDER)
        else:
            summary_msg = Message.system(SUMMARIZE_PLACEHOLDER)

        result = [*system_prefix, summary_msg, *tail_messages]

        # If the real summary ends up exceeding the budget (e.g. summarizer
        # produced a very long message), fall back to truncate_oldest.
        try:
            if count_tokens_impl(result, model, tools) <= self._budget:
                return result
        except Exception:
            # If counting fails for any reason, just return the summarized result.
            return result

        logger.warning(
            "Summarized context still exceeds budget; falling back to truncate_oldest."
        )
        flat_rest = [m for group in groups for m in group]
        return self._fit_truncate_oldest(system_prefix, flat_rest, tools, model)
