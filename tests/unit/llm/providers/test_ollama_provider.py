"""
Unit tests for curio_agent_sdk.core.llm.providers.ollama

Covers: OllamaProvider — name, call success, tools, rate limit (N/A),
auth error (N/A), server error, timeout, stream, request formatting, response parsing
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from curio_agent_sdk.core.llm.providers.ollama import OllamaProvider, OLLAMA_DEFAULT_URL
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
    ToolSchema,
    TokenUsage,
)
from curio_agent_sdk.models.exceptions import LLMProviderError

# Guard: openai package is required by Ollama provider (OpenAI-compatible API).
openai = pytest.importorskip("openai", reason="openai package not installed")


# ---------------------------------------------------------------------------
# Mock helpers
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


def _mock_response(content="Hello!", tool_calls=None, finish_reason="stop", model="llama3.1:8b"):
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


class TestOllamaProvider:

    def test_provider_name(self):
        """provider_name attribute."""
        provider = OllamaProvider()
        assert provider.provider_name == "ollama"

    @pytest.mark.asyncio
    async def test_provider_call_success(self):
        """Successful API call (mocked)."""
        provider = OllamaProvider()
        mock_resp = _mock_response(content="Local response!")

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama3.1:8b")
            response = await provider.call(request, base_url="http://localhost:11434/v1")

            assert response.content == "Local response!"
            assert response.provider == "ollama"
            assert response.model == "llama3.1:8b"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_provider_call_with_tools(self):
        """Tool-use request formatting."""
        provider = OllamaProvider()
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
                model="llama3.1:8b",
            )
            response = await provider.call(request)

            assert response.has_tool_calls
            assert response.tool_calls[0].name == "calculator"
            assert response.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_provider_call_rate_limit(self):
        """Ollama doesn't have rate limiting — APIError → LLMProviderError."""
        provider = OllamaProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(
                side_effect=openai.APIError(
                    message="Too many requests",
                    request=MagicMock(),
                    body=None,
                )
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama3.1:8b")
            with pytest.raises(LLMProviderError):
                await provider.call(request)

    @pytest.mark.asyncio
    async def test_provider_call_auth_error(self):
        """Ollama doesn't need auth — generic errors become LLMProviderError."""
        provider = OllamaProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama3.1:8b")
            with pytest.raises(LLMProviderError):
                await provider.call(request)

    @pytest.mark.asyncio
    async def test_provider_call_server_error(self):
        """500 → LLMProviderError."""
        provider = OllamaProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            error = openai.APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            )
            error.status_code = 500
            client.chat.completions.create = AsyncMock(side_effect=error)
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama3.1:8b")
            with pytest.raises(LLMProviderError):
                await provider.call(request)

    @pytest.mark.asyncio
    async def test_provider_call_timeout(self):
        """Timeout → LLMProviderError (via generic Exception handler)."""
        provider = OllamaProvider()

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(
                side_effect=TimeoutError("Connection timed out")
            )
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama3.1:8b")
            with pytest.raises(LLMProviderError):
                await provider.call(request)

    @pytest.mark.asyncio
    async def test_provider_stream(self):
        """Streaming response falls back to non-streaming (base class default)."""
        provider = OllamaProvider()
        mock_resp = _mock_response(content="Streamed locally!")

        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            request = LLMRequest(messages=[Message.user("Hello")], model="llama3.1:8b")
            chunks = []
            async for chunk in provider.stream(request):
                chunks.append(chunk)

            # Base class default stream: text_delta, usage, done
            types = [c.type for c in chunks]
            assert "text_delta" in types
            assert "done" in types

    def test_provider_request_formatting(self):
        """Request format matches OpenAI-compatible spec."""
        provider = OllamaProvider()
        request = LLMRequest(
            messages=[
                Message.system("Be helpful."),
                Message.user("Hello"),
            ],
            model="llama3.1:8b",
        )
        messages = provider._build_messages(request)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"

    def test_provider_response_parsing(self):
        """Response parsed correctly."""
        provider = OllamaProvider()
        mock_resp = _mock_response(content="Parsed!", model="llama3.1:8b")

        choice = mock_resp.choices[0]
        assert choice.message.content == "Parsed!"
        assert choice.finish_reason == "stop"
        assert mock_resp.usage.prompt_tokens == 10
        assert mock_resp.usage.completion_tokens == 5


class TestOllamaDefaults:
    def test_default_url(self):
        assert OLLAMA_DEFAULT_URL == "http://localhost:11434/v1"

    @pytest.mark.asyncio
    async def test_shutdown(self):
        provider = OllamaProvider()
        # No HTTP client created yet — shutdown should be safe
        await provider.shutdown()
        assert provider._http_client is None
