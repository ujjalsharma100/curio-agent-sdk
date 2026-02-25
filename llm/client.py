"""
LLMClient - the main interface for making LLM calls.

Combines tiered routing, provider dispatch, automatic failover,
usage tracking, and streaming into a single async interface.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Callable

from curio_agent_sdk.llm.router import TieredRouter, RouteResult
from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.providers.openai import OpenAIProvider
from curio_agent_sdk.llm.providers.anthropic import AnthropicProvider
from curio_agent_sdk.llm.providers.groq import GroqProvider
from curio_agent_sdk.llm.providers.ollama import OllamaProvider
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    TokenUsage,
)
from curio_agent_sdk.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMProviderError,
    NoAvailableModelError,
)

logger = logging.getLogger(__name__)

# Built-in provider registry
BUILTIN_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}

# Max retries for automatic failover
MAX_FAILOVER_RETRIES = 10


class LLMClient:
    """
    Unified async LLM client with routing, failover, and streaming.

    This is the primary interface for making LLM calls. It handles:
    - Provider selection via tiered routing
    - Automatic failover on rate limits / errors
    - Per-request API keys (no shared mutable state)
    - Streaming responses
    - Usage tracking via callbacks

    Example:
        client = LLMClient(
            router=TieredRouter(
                tier1=["groq:llama-3.1-8b-instant"],
                tier2=["openai:gpt-4o-mini"],
                tier3=["anthropic:claude-sonnet-4-6"],
            ),
        )

        response = await client.call(LLMRequest(
            messages=[Message.user("Hello!")],
            tier="tier2",
        ))
    """

    def __init__(
        self,
        router: TieredRouter | None = None,
        custom_providers: dict[str, type[LLMProvider]] | None = None,
        on_llm_usage: Callable | None = None,
    ):
        """
        Initialize the LLM client.

        Args:
            router: TieredRouter for provider/model selection. Auto-created from env if None.
            custom_providers: Additional provider classes (name -> class).
            on_llm_usage: Callback for usage tracking. Called with (provider, model, usage, latency_ms, error).
        """
        self.router = router or TieredRouter()
        self.on_llm_usage = on_llm_usage

        # Build provider class registry
        self._provider_classes: dict[str, type[LLMProvider]] = dict(BUILTIN_PROVIDERS)
        if custom_providers:
            self._provider_classes.update(custom_providers)

        # Provider instances are stateless (no shared API key), so we create one per class
        self._provider_instances: dict[str, LLMProvider] = {}
        for name, cls in self._provider_classes.items():
            try:
                self._provider_instances[name] = cls()
            except Exception:
                # Provider may not be installed (e.g., ollama not running)
                pass

    def _get_provider(self, name: str) -> LLMProvider:
        """Get or create a provider instance."""
        if name not in self._provider_instances:
            if name in self._provider_classes:
                self._provider_instances[name] = self._provider_classes[name]()
            else:
                # Fall back to OpenAI-compatible for unknown providers
                self._provider_instances[name] = OpenAIProvider()
        return self._provider_instances[name]

    async def call(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """
        Make an LLM call with automatic routing and failover.

        Args:
            request: The LLMRequest to send.
            run_id: Optional run ID for tracking.
            agent_id: Optional agent ID for tracking.

        Returns:
            LLMResponse from the selected provider.

        Raises:
            NoAvailableModelError: If no provider/model is available.
            LLMError: If the call fails after all retries.
        """
        excluded: list[str] = []
        last_error: Exception | None = None

        for attempt in range(MAX_FAILOVER_RETRIES):
            # Route the request
            route = self.router.route(
                tier=request.tier,
                provider=request.provider,
                model=request.model,
                excluded_models=excluded,
            )

            if not route:
                break

            try:
                provider = self._get_provider(route.provider)
                start = time.monotonic()

                # Make the actual call with per-request key
                req = self._apply_route(request, route)
                response = await provider.call(
                    req,
                    api_key=route.api_key,
                    base_url=route.base_url,
                )

                # Record success
                self.router.record_success(route.provider, route.key_name)

                # Track usage
                self._track_usage(route, response, run_id, agent_id)

                return response

            except LLMRateLimitError as e:
                logger.warning(
                    f"Rate limit for {route.provider}:{route.model} "
                    f"(key={route.key_name}), trying next... (attempt {attempt + 1})"
                )
                self.router.record_failure(route.provider, route.key_name, is_rate_limit=True)
                excluded.append(f"{route.provider}:{route.model}")
                last_error = e

            except LLMError as e:
                logger.warning(
                    f"LLM error for {route.provider}:{route.model}: {e} "
                    f"(attempt {attempt + 1})"
                )
                self.router.record_failure(route.provider, route.key_name)
                excluded.append(f"{route.provider}:{route.model}")
                last_error = e

        # All options exhausted
        tier_info = f" for tier {request.tier}" if request.tier else ""
        raise NoAvailableModelError(
            f"No available model{tier_info} after {len(excluded)} exclusions. "
            f"Excluded: {excluded}. Last error: {last_error}",
        )

    async def stream(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream an LLM response with automatic routing.

        Streaming does NOT auto-failover mid-stream. It routes once at the start.

        Args:
            request: The LLMRequest to send (stream flag is set automatically).
            run_id: Optional run ID for tracking.
            agent_id: Optional agent ID for tracking.

        Yields:
            LLMStreamChunk events.
        """
        route = self.router.route(
            tier=request.tier,
            provider=request.provider,
            model=request.model,
        )

        if not route:
            raise NoAvailableModelError(f"No available model for streaming request")

        provider = self._get_provider(route.provider)
        req = self._apply_route(request, route)
        req.stream = True

        try:
            async for chunk in provider.stream(req, api_key=route.api_key, base_url=route.base_url):
                yield chunk

            self.router.record_success(route.provider, route.key_name)

        except LLMRateLimitError:
            self.router.record_failure(route.provider, route.key_name, is_rate_limit=True)
            raise
        except LLMError:
            self.router.record_failure(route.provider, route.key_name)
            raise

    def _apply_route(self, request: LLMRequest, route: RouteResult) -> LLMRequest:
        """Create a copy of the request with the routed model applied."""
        return LLMRequest(
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            response_format=request.response_format,
            stop=request.stop,
            model=route.model,
            provider=route.provider,
            tier=request.tier,
            metadata=request.metadata,
        )

    def _track_usage(
        self,
        route: RouteResult,
        response: LLMResponse,
        run_id: str | None,
        agent_id: str | None,
    ):
        if self.on_llm_usage:
            try:
                self.on_llm_usage(
                    provider=route.provider,
                    model=route.model,
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                    error=response.error,
                    run_id=run_id,
                    agent_id=agent_id,
                )
            except Exception as e:
                logger.error(f"Usage tracking callback failed: {e}")

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return self.router.get_stats()
