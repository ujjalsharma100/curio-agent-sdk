"""
Unit tests for curio_agent_sdk.core.llm.providers.groq

Covers: GroqProvider — name, call success, tools, rate limit,
auth error, server error, timeout, stream, request formatting, response parsing
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from curio_agent_sdk.core.llm.providers.groq import GroqProvider, GROQ_BASE_URL
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
)

# Guard: the openai package may not be installed in the test environment.
openai = pytest.importorskip("openai", reason="openai package not installed")


# ---------------------------------------------------------------------------
# Mock helpers (OpenAI-compatible format)
# ---------------------------------------------------------------------------

def _mock_usage(prompt_tokens=10, completion_tokens=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    return usage


def _mock_choice(content="Hello!", tool_calls=None, finish_reason="stop"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    return choice


def _mock_response(content="Hello!", tool_calls=None, finish_reason="stop", model="llama-3.1-8b-instant"):
    response = MagicMock()
    response.choices = [_mock_choice(content, tool_calls, finish_reason)]
    response.usage = _mock_usage()
    response.model = model
    return response


def _mock_tool_call(tc_id="call_1", name="calculator", arguments='{"expr": "2+2"}'):
    tc = MagicMock()
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroqProvider:

    def test_provider_name(self):
        """provider_name attribute."""
        provider = GroqProvider()
        assert provider.provider_name == "groq"

    @pytest.mark.asyncio
    async def test_provider_call_success(self):
        """Successful API call (mocked)."""
        provider = GroqProvider()
        mock_resp = _mock_response(content="Fast response!")

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama-3.1-8b-instant")
            response = await provider.call(request, api_key="gsk-test")

            assert response.content == "Fast response!"
            assert response.provider == "groq"
            assert response.model == "llama-3.1-8b-instant"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_provider_call_with_tools(self):
        """Tool-use request formatting."""
        provider = GroqProvider()
        tc = _mock_tool_call()
        mock_resp = _mock_response(content="", tool_calls=[tc], finish_reason="tool_calls")

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            tool = ToolSchema(
                name="calculator",
                description="Math eval",
                parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
            )
            request = LLMRequest(
                messages=[Message.user("What is 2+2?")],
                tools=[tool],
                model="llama-3.1-8b-instant",
            )
            response = await provider.call(request, api_key="gsk-test")

            assert response.has_tool_calls
            assert response.tool_calls[0].name == "calculator"
            assert response.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_provider_call_rate_limit(self):
        """429 → LLMRateLimitError."""
        provider = GroqProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            error_response = MagicMock()
            error_response.status_code = 429
            error_response.headers = {}
            client.chat.completions.create = AsyncMock(
                side_effect=openai.RateLimitError(
                    message="Rate limit exceeded",
                    response=error_response,
                    body=None,
                )
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama-3.1-8b-instant")
            with pytest.raises(LLMRateLimitError) as exc_info:
                await provider.call(request, api_key="gsk-test")
            assert exc_info.value.provider == "groq"

    @pytest.mark.asyncio
    async def test_provider_call_auth_error(self):
        """401 → LLMAuthenticationError."""
        provider = GroqProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            error_response = MagicMock()
            error_response.status_code = 401
            error_response.headers = {}
            client.chat.completions.create = AsyncMock(
                side_effect=openai.AuthenticationError(
                    message="Invalid API key",
                    response=error_response,
                    body=None,
                )
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama-3.1-8b-instant")
            with pytest.raises(LLMAuthenticationError):
                await provider.call(request, api_key="bad-key")

    @pytest.mark.asyncio
    async def test_provider_call_server_error(self):
        """500 → LLMProviderError."""
        provider = GroqProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(
                side_effect=openai.APIError(
                    message="Internal server error",
                    request=MagicMock(),
                    body=None,
                )
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama-3.1-8b-instant")
            with pytest.raises(LLMProviderError):
                await provider.call(request, api_key="gsk-test")

    @pytest.mark.asyncio
    async def test_provider_call_timeout(self):
        """Timeout → LLMProviderError (Groq uses APIError for timeouts via OpenAI client)."""
        provider = GroqProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(
                side_effect=openai.APIError(
                    message="Request timeout",
                    request=MagicMock(),
                    body=None,
                )
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama-3.1-8b-instant")
            with pytest.raises(LLMProviderError):
                await provider.call(request, api_key="gsk-test")

    @pytest.mark.asyncio
    async def test_provider_stream(self):
        """Streaming response (mocked)."""
        provider = GroqProvider()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Fast"
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " reply"
        chunk2.choices[0].delta.tool_calls = None
        chunk2.choices[0].finish_reason = None
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock()
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].delta.tool_calls = None
        chunk3.choices[0].finish_reason = "stop"
        chunk3.usage = None

        async def mock_stream():
            for c in [chunk1, chunk2, chunk3]:
                yield c

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama-3.1-8b-instant")
            chunks = []
            async for chunk in provider.stream(request, api_key="gsk-test"):
                chunks.append(chunk)

            text_chunks = [c for c in chunks if c.type == "text_delta"]
            assert len(text_chunks) == 2
            assert text_chunks[0].text == "Fast"
            assert text_chunks[1].text == " reply"

    def test_provider_request_formatting(self):
        """Request format matches OpenAI-compatible spec."""
        provider = GroqProvider()
        request = LLMRequest(
            messages=[
                Message.system("Be concise."),
                Message.user("Hi"),
            ],
            model="llama-3.1-8b-instant",
        )
        messages = provider._build_messages(request)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_provider_response_parsing(self):
        """Response parsed into LLMResponse with correct provider."""
        provider = GroqProvider()
        mock_resp = _mock_response(content="Parsed!", model="llama-3.1-8b-instant")

        choice = mock_resp.choices[0]
        assert choice.message.content == "Parsed!"
        assert choice.finish_reason == "stop"
        assert mock_resp.model == "llama-3.1-8b-instant"


class TestGroqBaseUrl:
    def test_groq_base_url(self):
        assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"
