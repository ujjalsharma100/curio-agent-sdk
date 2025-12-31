"""
Ollama provider implementation for local models.
"""

import logging
from typing import Optional

from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama library not installed. Install with: pip install ollama")


class OllamaProvider(LLMProvider):
    """
    Ollama provider implementation for local models.

    Supports running models locally through Ollama.

    Example:
        >>> config = LLMConfig(
        ...     provider="ollama",
        ...     model="llama2",
        ...     base_url="http://localhost:11434",
        ... )
        >>> provider = OllamaProvider(config)
        >>> response = provider.call("Hello, world!")
        >>> print(response.content)
    """

    def _initialize_client(self) -> None:
        """Initialize Ollama client."""
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama library not installed. Install with: pip install ollama"
            )

        base_url = self.config.base_url or "http://localhost:11434"
        self.client = ollama.Client(host=base_url)

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Call Ollama API.

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
                "prompt": prompt,
                "options": {
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                },
            }

            # Add optional parameters to options
            if "top_p" in kwargs:
                params["options"]["top_p"] = kwargs["top_p"]

            response = self.client.generate(**params)

            return LLMResponse(
                content=response["response"],
                provider="ollama",
                model=response["model"],
                usage={
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                },
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"Ollama API error: {error_str}")

            return LLMResponse(
                content="",
                provider="ollama",
                model=model,
                error=error_str,
            )
