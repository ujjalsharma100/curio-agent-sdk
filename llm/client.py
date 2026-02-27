"""
LLMClient - the main interface for making LLM calls.

Combines tiered routing, provider dispatch, automatic failover,
usage tracking, request deduplication, and streaming into a single async interface.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

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
)
from curio_agent_sdk.models.exceptions import (
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


@dataclass
class _DedupeEntry:
    """Cached LLM response with expiry."""
    response: LLMResponse
    expires_at: float


class LLMClient:
    """
    Unified async LLM client with routing, failover, and streaming.

    This is the primary interface for making LLM calls. It handles:
    - Provider selection via tiered routing
    - Automatic failover on rate limits / errors with backoff
    - Per-request API keys (no shared mutable state)
    - Streaming responses

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
        dedup_enabled: bool = False,
        dedup_ttl: float = 30.0,
    ):
        """
        Initialize the LLM client.

        Args:
            router: TieredRouter for provider/model selection. Auto-created from env if None.
            custom_providers: Additional provider classes (name -> class).
            dedup_enabled: Enable request deduplication cache. Identical LLM
                calls within dedup_ttl seconds return cached results.
            dedup_ttl: Time-to-live (seconds) for deduplication cache entries.
        """
        self.router = router or TieredRouter()

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

        # Request deduplication cache
        self._dedup_enabled = dedup_enabled
        self._dedup_ttl = dedup_ttl
        self._dedup_cache: dict[str, _DedupeEntry] = {}

    def _dedup_key(self, request: LLMRequest) -> str:
        """Generate a deterministic hash for an LLM request for dedup purposes."""
        msg_data = []
        for m in request.messages:
            msg_data.append({"role": m.role, "content": m.content})
        payload = json.dumps({
            "messages": msg_data,
            "model": request.model or "",
            "provider": request.provider or "",
            "tier": request.tier or "",
            "tools": [t.get("function", {}).get("name", "") for t in (request.tools or [])],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_dedup_cached(self, request: LLMRequest) -> LLMResponse | None:
        """Return cached response if within TTL, else None."""
        if not self._dedup_enabled:
            return None
        key = self._dedup_key(request)
        entry = self._dedup_cache.get(key)
        if entry is not None:
            if time.monotonic() < entry.expires_at:
                logger.debug("Dedup cache hit for LLM request")
                return entry.response
            del self._dedup_cache[key]
        return None

    def _set_dedup_cached(self, request: LLMRequest, response: LLMResponse) -> None:
        """Store a response in the dedup cache."""
        if not self._dedup_enabled:
            return
        key = self._dedup_key(request)
        self._dedup_cache[key] = _DedupeEntry(
            response=response,
            expires_at=time.monotonic() + self._dedup_ttl,
        )

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
        # Check dedup cache
        cached = self._get_dedup_cached(request)
        if cached is not None:
            return cached

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

                # Cache for dedup
                self._set_dedup_cached(request, response)

                return response

            except LLMRateLimitError as e:
                # Optional exponential backoff before failing over to the next model
                base_delay = getattr(self.router, "retry_delay", 1.0)
                max_delay = getattr(self.router, "max_retry_delay", 30.0)
                retry_on_rate_limit = getattr(self.router, "retry_on_rate_limit", True)
                delay = 0.0
                if retry_on_rate_limit:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if e.retry_after:
                        delay = max(delay, e.retry_after)

                logger.warning(
                    f"Rate limit for {route.provider}:{route.model} "
                    f"(key={route.key_name}), trying next... (attempt {attempt + 1}, backoff={delay:.1f}s)"
                )

                if delay > 0:
                    await asyncio.sleep(delay)

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

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return self.router.get_stats()
