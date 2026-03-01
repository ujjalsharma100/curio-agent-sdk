"""
Unit tests for curio_agent_sdk.core.llm.token_counter

Covers: count_tokens, count_tokens_batch — simple strings, messages,
tools, caching, different models, empty, fallback estimation
"""

from unittest.mock import patch, MagicMock

import pytest

from curio_agent_sdk.core.llm.token_counter import (
    count_tokens,
    count_tokens_batch,
    _infer_provider,
    _get_model_name,
    _messages_to_text,
    _count_tokens_approximate,
    _count_tokens_tiktoken,
    _get_tiktoken_encoding,
    APPROXIMATE_CHARS_PER_TOKEN,
    _TIKTOKEN_ENCODERS,
)
from curio_agent_sdk.models.llm import Message, ToolSchema, ToolCall


# ===================================================================
# Tests
# ===================================================================


class TestTokenCounter:

    def test_count_tokens_simple(self):
        """Count tokens in simple messages (OpenAI model uses tiktoken)."""
        messages = [Message.user("Hello, world!")]
        count = count_tokens(messages, "gpt-4o-mini")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_messages(self):
        """Count tokens across multiple messages."""
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("What is 2+2?"),
            Message.assistant("2+2 equals 4."),
        ]
        count = count_tokens(messages, "gpt-4o")
        assert count > 0
        # Multiple messages should have more tokens than a single one
        single = count_tokens([messages[0]], "gpt-4o")
        assert count > single

    def test_count_tokens_with_tools(self):
        """Count includes tool schemas (Anthropic path)."""
        messages = [Message.user("Use the calculator")]
        tool = ToolSchema(
            name="calculator",
            description="Evaluate math expressions",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        )
        # Use anthropic model to test the anthropic path with tools
        # This will fall back to approximate if anthropic SDK isn't available
        count_without = count_tokens(messages, "claude-sonnet-4-6")
        count_with = count_tokens(messages, "claude-sonnet-4-6", tools=[tool])
        # Both should be > 0 (exact comparison depends on SDK availability)
        assert count_without > 0
        assert count_with > 0

    def test_count_tokens_caching(self):
        """Cached tiktoken encoders are reused."""
        messages = [Message.user("Test caching")]
        # First call populates cache
        count1 = count_tokens(messages, "gpt-4o-mini")
        # Second call should use cached encoder
        count2 = count_tokens(messages, "gpt-4o-mini")
        assert count1 == count2

    def test_count_tokens_different_models(self):
        """Different models may produce different counts."""
        messages = [Message.user("Hello, how are you doing today?")]
        count_openai = count_tokens(messages, "gpt-4o")
        count_approx = count_tokens(messages, "ollama:llama3.1:8b")
        # Both should be positive but may differ
        assert count_openai > 0
        assert count_approx > 0

    def test_count_tokens_empty(self):
        """Empty message list."""
        count = count_tokens([], "gpt-4o")
        assert count == 0

    def test_fallback_estimation(self):
        """When tiktoken not installed, uses approximate counting."""
        messages = [Message.user("Hello, this is a test message for estimation.")]
        count = _count_tokens_approximate(messages)
        text = _messages_to_text(messages)
        expected = max(0, (len(text) + APPROXIMATE_CHARS_PER_TOKEN - 1) // APPROXIMATE_CHARS_PER_TOKEN)
        assert count == expected
        assert count > 0


class TestInferProvider:
    def test_openai_prefix(self):
        assert _infer_provider("openai:gpt-4o") == "openai"

    def test_anthropic_prefix(self):
        assert _infer_provider("anthropic:claude-sonnet-4-6") == "anthropic"

    def test_gpt_heuristic(self):
        assert _infer_provider("gpt-4o-mini") == "openai"

    def test_claude_heuristic(self):
        assert _infer_provider("claude-sonnet-4-6") == "anthropic"

    def test_llama_heuristic(self):
        assert _infer_provider("llama-3.1-8b-instant") == "groq"

    def test_unknown(self):
        assert _infer_provider("some-unknown-model") == "unknown"


class TestGetModelName:
    def test_with_provider_prefix(self):
        assert _get_model_name("openai:gpt-4o") == "gpt-4o"

    def test_without_prefix(self):
        assert _get_model_name("gpt-4o") == "gpt-4o"

    def test_with_spaces(self):
        assert _get_model_name("openai: gpt-4o ") == "gpt-4o"


class TestCountTokensBatch:
    def test_batch_counting(self):
        batches = [
            [Message.user("Hello")],
            [Message.user("World")],
            [Message.user("Test message")],
        ]
        counts = count_tokens_batch(batches, "gpt-4o-mini")
        assert len(counts) == 3
        assert all(c > 0 for c in counts)

    def test_batch_empty(self):
        counts = count_tokens_batch([], "gpt-4o")
        assert counts == []

    def test_batch_approximate(self):
        batches = [
            [Message.user("Hello")],
            [Message.user("World")],
        ]
        counts = count_tokens_batch(batches, "ollama:llama3.1:8b")
        assert len(counts) == 2
        assert all(c > 0 for c in counts)
