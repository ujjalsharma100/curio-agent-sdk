"""
Anthropic provider implementation.
"""

import logging
from typing import Optional

from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not installed. Install with: pip install anthropic")


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation.

    Supports Claude models (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.).

    Example:
        >>> config = LLMConfig(
        ...     provider="anthropic",
        ...     api_key="sk-ant-...",
        ...     model="claude-3-sonnet-20240229",
        ... )
        >>> provider = AnthropicProvider(config)
        >>> response = provider.call("Hello, world!")
        >>> print(response.content)
    """

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        if not self.config.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.Anthropic(api_key=self.config.api_key)

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Call Anthropic API.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (model, max_tokens, temperature, etc.)

        Returns:
            LLMResponse with generated content
        """
        model = kwargs.get("model", self.config.model)

        try:
            params = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add optional parameters
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "stop" in kwargs:
                params["stop_sequences"] = kwargs["stop"]

            response = self.client.messages.create(**params)

            return LLMResponse(
                content=response.content[0].text,
                provider="anthropic",
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"Anthropic API error: {error_str}")

            # Check for rate limit
            is_rate_limit = (
                "rate limit" in error_str.lower() or
                "429" in error_str or
                hasattr(e, 'status_code') and e.status_code == 429
            )

            return LLMResponse(
                content="",
                provider="anthropic",
                model=model,
                error=error_str,
                usage={"is_rate_limit": is_rate_limit} if is_rate_limit else None,
            )
