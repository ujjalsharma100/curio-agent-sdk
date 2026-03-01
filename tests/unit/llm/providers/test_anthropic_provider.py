"""
Unit tests for curio_agent_sdk.core.llm.providers.anthropic

Covers: AnthropicProvider — name, call success, tools, rate limit,
auth error, server error, timeout, stream, request formatting, response parsing
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from curio_agent_sdk.core.llm.providers.anthropic import AnthropicProvider
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
    ToolSchema,
    TokenUsage,
)
from curio_agent_sdk.models.exceptions import (
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMTimeoutError,
)

# Guard: the anthropic package may not be installed in the test environment.
anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_text_block(text="Hello!"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _mock_tool_use_block(block_id="tool_1", name="calculator", input_data=None):
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = input_data or {"expr": "2+2"}
    return block


def _mock_usage(input_tokens=10, output_tokens=5):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    return usage


def _mock_response(content_blocks=None, stop_reason="end_turn", model="claude-sonnet-4-6"):
    response = MagicMock()
    response.content = content_blocks or [_mock_text_block()]
    response.stop_reason = stop_reason
    response.model = model
    response.usage = _mock_usage()
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:

    def test_provider_name(self):
        """provider_name attribute."""
        provider = AnthropicProvider()
        assert provider.provider_name == "anthropic"

    @pytest.mark.asyncio
    async def test_provider_call_success(self):
        """Successful API call (mocked)."""
        provider = AnthropicProvider()
        mock_resp = _mock_response(content_blocks=[_mock_text_block("Hi there!")])

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            client.messages.create = AsyncMock(return_value=mock_resp)
            mock_get_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="claude-sonnet-4-6")
            response = await provider.call(request, api_key="sk-test")

            assert response.content == "Hi there!"
            assert response.provider == "anthropic"
            assert response.model == "claude-sonnet-4-6"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_provider_call_with_tools(self):
        """Tool-use request formatting."""
        provider = AnthropicProvider()
        mock_resp = _mock_response(
            content_blocks=[_mock_tool_use_block()],
            stop_reason="tool_use",
        )

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            client.messages.create = AsyncMock(return_value=mock_resp)
            mock_get_client.return_value = client

            tool = ToolSchema(
                name="calculator",
                description="Math eval",
                parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
            )
            request = LLMRequest(
                messages=[Message.user("What is 2+2?")],
                tools=[tool],
                model="claude-sonnet-4-6",
            )
            response = await provider.call(request, api_key="sk-test")

            assert response.has_tool_calls
            assert response.tool_calls[0].name == "calculator"
            assert response.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_provider_call_rate_limit(self):
        """429 → LLMRateLimitError."""
        provider = AnthropicProvider()

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            error_response = MagicMock()
            error_response.status_code = 429
            error_response.headers = {}
            client.messages.create = AsyncMock(
                side_effect=anthropic.RateLimitError(
                    message="Rate limit exceeded",
                    response=error_response,
                    body=None,
                )
            )
            mock_get_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="claude-sonnet-4-6")
            with pytest.raises(LLMRateLimitError) as exc_info:
                await provider.call(request, api_key="sk-test")
            assert exc_info.value.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_provider_call_auth_error(self):
        """401 → LLMAuthenticationError."""
        provider = AnthropicProvider()

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            error_response = MagicMock()
            error_response.status_code = 401
            error_response.headers = {}
            client.messages.create = AsyncMock(
                side_effect=anthropic.AuthenticationError(
                    message="Invalid API key",
                    response=error_response,
                    body=None,
                )
            )
            mock_get_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="claude-sonnet-4-6")
            with pytest.raises(LLMAuthenticationError):
                await provider.call(request, api_key="bad-key")

    @pytest.mark.asyncio
    async def test_provider_call_server_error(self):
        """500 → LLMProviderError."""
        provider = AnthropicProvider()

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            client.messages.create = AsyncMock(
                side_effect=anthropic.APIError(
                    message="Internal server error",
                    request=MagicMock(),
                    body=None,
                )
            )
            mock_get_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="claude-sonnet-4-6")
            with pytest.raises(LLMProviderError):
                await provider.call(request, api_key="sk-test")

    @pytest.mark.asyncio
    async def test_provider_call_timeout(self):
        """Timeout → LLMTimeoutError."""
        provider = AnthropicProvider()

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            client.messages.create = AsyncMock(
                side_effect=anthropic.APITimeoutError(request=MagicMock())
            )
            mock_get_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="claude-sonnet-4-6")
            with pytest.raises(LLMTimeoutError):
                await provider.call(request, api_key="sk-test")

    @pytest.mark.asyncio
    async def test_provider_stream(self):
        """Streaming response (mocked)."""
        provider = AnthropicProvider()

        # Create mock stream events
        event1 = MagicMock()
        event1.type = "message_start"
        event1.message = MagicMock()
        event1.message.usage = MagicMock()
        event1.message.usage.input_tokens = 10
        event1.message.usage.cache_read_input_tokens = 0
        event1.message.usage.cache_creation_input_tokens = 0

        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta = MagicMock()
        event2.delta.type = "text_delta"
        event2.delta.text = "Hello world"

        event3 = MagicMock()
        event3.type = "message_delta"
        event3.delta = MagicMock()
        event3.delta.stop_reason = "end_turn"
        event3.usage = MagicMock()
        event3.usage.output_tokens = 5

        async def mock_events():
            for e in [event1, event2, event3]:
                yield e

        with patch.object(provider, "_get_client") as mock_get_client:
            client = AsyncMock()
            stream_ctx = AsyncMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
            stream_ctx.__aexit__ = AsyncMock(return_value=False)
            stream_ctx.__aiter__ = lambda self: mock_events()
            client.messages.stream = MagicMock(return_value=stream_ctx)
            mock_get_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="claude-sonnet-4-6")
            chunks = []
            async for chunk in provider.stream(request, api_key="sk-test"):
                chunks.append(chunk)

            types = [c.type for c in chunks]
            assert "usage" in types
            assert "text_delta" in types
            assert "done" in types

    def test_provider_request_formatting(self):
        """Request format matches Anthropic API spec."""
        provider = AnthropicProvider()
        request = LLMRequest(
            messages=[
                Message.system("You are helpful."),
                Message.user("Hi"),
                Message.assistant("Hello!"),
            ],
            model="claude-sonnet-4-6",
        )
        params = provider._build_params(request, "claude-sonnet-4-6")

        assert params["model"] == "claude-sonnet-4-6"
        assert params["system"] == "You are helpful."
        # System message should not be in messages
        assert all(m["role"] != "system" for m in params["messages"])
        assert len(params["messages"]) == 2

    def test_provider_response_parsing(self):
        """Response parsed into LLMResponse."""
        provider = AnthropicProvider()
        mock_resp = _mock_response(
            content_blocks=[_mock_text_block("Parsed!")],
            model="claude-sonnet-4-6",
        )

        result = provider._parse_response(mock_resp, "claude-sonnet-4-6", 42)

        assert isinstance(result, LLMResponse)
        assert result.content == "Parsed!"
        assert result.model == "claude-sonnet-4-6"
        assert result.provider == "anthropic"
        assert result.latency_ms == 42
        assert result.finish_reason == "stop"  # end_turn maps to stop
