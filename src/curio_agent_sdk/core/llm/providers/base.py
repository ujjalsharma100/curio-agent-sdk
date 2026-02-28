"""
Abstract base class for async LLM providers.

All providers implement message-based APIs with native tool calling support.
API keys are passed per-request to avoid race conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator
import logging

from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for async LLM providers.

    Providers translate our provider-agnostic LLMRequest/LLMResponse models
    into provider-specific API calls. They must support:
    - Message-based conversation (system/user/assistant/tool roles)
    - Native tool/function calling
    - Streaming responses
    - Per-request API key (no shared mutable state)
    """

    provider_name: str = "base"

    @abstractmethod
    async def call(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMResponse:
        """
        Make a non-streaming call to the LLM.

        Args:
            request: The provider-agnostic request.
            api_key: API key for this specific call (avoids shared state).
            base_url: Optional base URL override.

        Returns:
            LLMResponse with the assistant's message.
        """
        ...

    async def stream(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Make a streaming call to the LLM.

        Default implementation falls back to non-streaming call.

        Args:
            request: The provider-agnostic request.
            api_key: API key for this specific call.
            base_url: Optional base URL override.

        Yields:
            LLMStreamChunk events.
        """
        # Default: fall back to non-streaming
        response = await self.call(request, api_key=api_key, base_url=base_url)
        if response.content:
            yield LLMStreamChunk(type="text_delta", text=response.content)
        if response.usage:
            yield LLMStreamChunk(type="usage", usage=response.usage)
        yield LLMStreamChunk(
            type="done",
            finish_reason=response.finish_reason,
            usage=response.usage,
        )
