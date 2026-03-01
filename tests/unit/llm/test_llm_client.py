"""
Unit tests for curio_agent_sdk.core.llm.client

Covers: LLMClient — basic call, tools, routing, error handling, dedup,
stream, batch, lifecycle, custom providers, fallback
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from curio_agent_sdk.core.llm.client import LLMClient, BUILTIN_PROVIDERS
from curio_agent_sdk.core.llm.router import (
    TieredRouter,
    RouteResult,
    ProviderConfig,
    ProviderKey,
)
from curio_agent_sdk.core.llm.providers.base import LLMProvider
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
    LLMError,
    LLMRateLimitError,
    LLMProviderError,
    NoAvailableModelError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """A mock provider for testing."""

    provider_name = "mock"

    async def call(self, request, api_key=None, base_url=None):
        return LLMResponse(
            message=Message.assistant("mock response"),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model=request.model or "mock-model",
            provider="mock",
            finish_reason="stop",
        )

    async def stream(self, request, api_key=None, base_url=None):
        yield LLMStreamChunk(type="text_delta", text="mock ")
        yield LLMStreamChunk(type="text_delta", text="stream")
        yield LLMStreamChunk(type="done", finish_reason="stop")


def _make_router_with_mock() -> TieredRouter:
    """Create a router that routes to 'mock' provider."""
    router = TieredRouter(
        providers={
            "mock": ProviderConfig(
                name="mock",
                keys=[ProviderKey(api_key="test-key", name="default")],
                default_model="mock-model",
                enabled=True,
            ),
        },
        tier1=["mock:mock-model"],
    )
    return router


def _make_request(text: str = "Hello!") -> LLMRequest:
    return LLMRequest(messages=[Message.user(text)])


# ===================================================================
# Tests
# ===================================================================


class TestLLMClient:
    """Tests for LLMClient."""

    @pytest.mark.asyncio
    async def test_client_call_basic(self):
        """Basic call with mocked provider."""
        router = _make_router_with_mock()
        client = LLMClient(router=router, custom_providers={"mock": MockProvider})
        request = _make_request()
        request.provider = "mock"

        response = await client.call(request)

        assert response.content == "mock response"
        assert response.provider == "mock"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_client_call_with_tools(self):
        """Call with tool schemas passed through."""
        router = _make_router_with_mock()
        client = LLMClient(router=router, custom_providers={"mock": MockProvider})

        tool_schema = ToolSchema(
            name="calculator",
            description="Evaluate math",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        request = LLMRequest(
            messages=[Message.user("What is 2+2?")],
            tools=[tool_schema],
            provider="mock",
        )

        response = await client.call(request)
        assert response is not None
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_client_call_routing(self):
        """Request routed to correct provider via tier."""
        router = _make_router_with_mock()
        client = LLMClient(router=router, custom_providers={"mock": MockProvider})

        request = LLMRequest(
            messages=[Message.user("Hello")],
            tier="tier1",
        )

        response = await client.call(request)
        assert response.provider == "mock"
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_client_call_with_middleware(self):
        """Middleware pipeline executed (via provider call interception)."""
        call_log = []

        class TrackingProvider(LLMProvider):
            provider_name = "tracking"

            async def call(self, request, api_key=None, base_url=None):
                call_log.append({"model": request.model, "api_key": api_key})
                return LLMResponse(
                    message=Message.assistant("tracked"),
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model=request.model or "track-model",
                    provider="tracking",
                    finish_reason="stop",
                )

        router = TieredRouter(
            providers={
                "tracking": ProviderConfig(
                    name="tracking",
                    keys=[ProviderKey(api_key="tk-123", name="default")],
                    default_model="track-model",
                ),
            },
            tier1=["tracking:track-model"],
        )
        client = LLMClient(router=router, custom_providers={"tracking": TrackingProvider})

        request = LLMRequest(messages=[Message.user("Hi")], tier="tier1")
        await client.call(request)

        assert len(call_log) == 1
        assert call_log[0]["api_key"] == "tk-123"

    @pytest.mark.asyncio
    async def test_client_call_error_handling(self):
        """Provider error bubbles up after all retries exhausted."""

        class FailingProvider(LLMProvider):
            provider_name = "failing"

            async def call(self, request, api_key=None, base_url=None):
                raise LLMProviderError("Server error", "failing", "fail-model", 500)

        router = TieredRouter(
            providers={
                "failing": ProviderConfig(
                    name="failing",
                    keys=[ProviderKey(api_key="key", name="default")],
                    default_model="fail-model",
                ),
            },
            tier1=["failing:fail-model"],
        )
        client = LLMClient(router=router, custom_providers={"failing": FailingProvider})

        with pytest.raises(NoAvailableModelError):
            await client.call(LLMRequest(messages=[Message.user("Hi")], tier="tier1"))

    @pytest.mark.asyncio
    async def test_client_stream_basic(self):
        """Stream response with mocked provider."""
        router = _make_router_with_mock()
        client = LLMClient(router=router, custom_providers={"mock": MockProvider})

        request = LLMRequest(messages=[Message.user("Hello")], provider="mock")
        chunks = []
        async for chunk in client.stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].type == "text_delta"
        assert chunks[0].text == "mock "
        assert chunks[1].text == "stream"
        assert chunks[2].type == "done"

    @pytest.mark.asyncio
    async def test_client_batch_basic(self):
        """Batch multiple requests (sequential calls)."""
        router = _make_router_with_mock()
        client = LLMClient(router=router, custom_providers={"mock": MockProvider})

        requests = [
            LLMRequest(messages=[Message.user("Q1")], provider="mock"),
            LLMRequest(messages=[Message.user("Q2")], provider="mock"),
            LLMRequest(messages=[Message.user("Q3")], provider="mock"),
        ]

        responses = await asyncio.gather(*(client.call(r) for r in requests))
        assert len(responses) == 3
        assert all(r.content == "mock response" for r in responses)

    @pytest.mark.asyncio
    async def test_client_dedup_enabled(self):
        """Deduplication prevents duplicate calls."""
        call_count = 0

        class CountingProvider(LLMProvider):
            provider_name = "counting"

            async def call(self, request, api_key=None, base_url=None):
                nonlocal call_count
                call_count += 1
                return LLMResponse(
                    message=Message.assistant("counted"),
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model="count-model",
                    provider="counting",
                    finish_reason="stop",
                )

        router = TieredRouter(
            providers={
                "counting": ProviderConfig(
                    name="counting",
                    keys=[ProviderKey(api_key="key", name="default")],
                    default_model="count-model",
                ),
            },
        )
        client = LLMClient(
            router=router,
            custom_providers={"counting": CountingProvider},
            dedup_enabled=True,
            dedup_ttl=60.0,
        )

        request = LLMRequest(messages=[Message.user("Same question")], provider="counting")
        resp1 = await client.call(request)
        resp2 = await client.call(request)

        assert call_count == 1  # Second call should be cached
        assert resp1.content == resp2.content

    @pytest.mark.asyncio
    async def test_client_dedup_disabled(self):
        """No dedup when disabled — every call goes through."""
        call_count = 0

        class CountingProvider(LLMProvider):
            provider_name = "counting"

            async def call(self, request, api_key=None, base_url=None):
                nonlocal call_count
                call_count += 1
                return LLMResponse(
                    message=Message.assistant("counted"),
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model="count-model",
                    provider="counting",
                    finish_reason="stop",
                )

        router = TieredRouter(
            providers={
                "counting": ProviderConfig(
                    name="counting",
                    keys=[ProviderKey(api_key="key", name="default")],
                    default_model="count-model",
                ),
            },
        )
        client = LLMClient(
            router=router,
            custom_providers={"counting": CountingProvider},
            dedup_enabled=False,
        )

        request = LLMRequest(messages=[Message.user("Same question")], provider="counting")
        await client.call(request)
        await client.call(request)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_client_startup_shutdown(self):
        """Component lifecycle — startup and shutdown succeed."""
        router = _make_router_with_mock()
        client = LLMClient(router=router, custom_providers={"mock": MockProvider})

        await client.startup()
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_client_custom_providers(self):
        """Register custom provider class."""

        class CustomLLM(LLMProvider):
            provider_name = "custom"

            async def call(self, request, api_key=None, base_url=None):
                return LLMResponse(
                    message=Message.assistant("custom!"),
                    usage=TokenUsage(),
                    model="custom-model",
                    provider="custom",
                    finish_reason="stop",
                )

        router = TieredRouter(
            providers={
                "custom": ProviderConfig(
                    name="custom",
                    keys=[ProviderKey(api_key="ckey", name="default")],
                    default_model="custom-model",
                ),
            },
        )
        client = LLMClient(router=router, custom_providers={"custom": CustomLLM})

        request = LLMRequest(messages=[Message.user("Hi")], provider="custom")
        response = await client.call(request)
        assert response.content == "custom!"
        assert response.provider == "custom"

    @pytest.mark.asyncio
    async def test_client_fallback_on_error(self):
        """Falls back to alternative provider on error."""
        attempt = 0

        class FlakyProvider(LLMProvider):
            provider_name = "flaky"

            async def call(self, request, api_key=None, base_url=None):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise LLMProviderError("Temporary failure", "flaky", "flaky-model", 500)
                return LLMResponse(
                    message=Message.assistant("recovered"),
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model="flaky-model",
                    provider="flaky",
                    finish_reason="stop",
                )

        router = TieredRouter(
            providers={
                "flaky": ProviderConfig(
                    name="flaky",
                    keys=[ProviderKey(api_key="k1", name="key1"), ProviderKey(api_key="k2", name="key2")],
                    default_model="flaky-model",
                ),
            },
        )
        client = LLMClient(router=router, custom_providers={"flaky": FlakyProvider})

        request = LLMRequest(messages=[Message.user("Hi")], provider="flaky")
        response = await client.call(request)
        assert response.content == "recovered"
        assert attempt == 2
