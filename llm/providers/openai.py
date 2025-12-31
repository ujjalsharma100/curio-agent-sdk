"""
OpenAI provider implementation.
"""

import logging
from typing import Optional

from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Install with: pip install openai")


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.

    Supports GPT-3.5, GPT-4, and other OpenAI models.

    Example:
        >>> config = LLMConfig(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ... )
        >>> provider = OpenAIProvider(config)
        >>> response = provider.call("Hello, world!")
        >>> print(response.content)
    """

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        if not self.config.api_key:
            raise ValueError("OpenAI API key not provided")

        # Support custom base_url for OpenAI-compatible endpoints (e.g., on-prem deployments)
        client_kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        self.client = openai.OpenAI(**client_kwargs)

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Call OpenAI API.

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
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }

            # Add optional parameters
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "stop" in kwargs:
                params["stop"] = kwargs["stop"]

            response = self.client.chat.completions.create(**params)

            return LLMResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=response.model,
                usage=response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None,
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"OpenAI API error: {error_str}")

            # Check for rate limit
            is_rate_limit = (
                "rate limit" in error_str.lower() or
                "429" in error_str or
                hasattr(e, 'status_code') and e.status_code == 429
            )

            return LLMResponse(
                content="",
                provider="openai",
                model=model,
                error=error_str,
                usage={"is_rate_limit": is_rate_limit} if is_rate_limit else None,
            )
