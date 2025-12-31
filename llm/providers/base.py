"""
Abstract base class for LLM providers.

All provider implementations must inherit from LLMProvider and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from curio_agent_sdk.llm.models import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class and
    implement the _initialize_client() and call() methods.

    Example:
        class MyProvider(LLMProvider):
            def _initialize_client(self):
                self.client = MyClient(self.config.api_key)

            def call(self, prompt: str, **kwargs) -> LLMResponse:
                response = self.client.generate(prompt)
                return LLMResponse(
                    content=response.text,
                    provider="myprovider",
                    model=kwargs.get("model", "default"),
                )
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the provider with configuration.

        Args:
            config: LLMConfig containing provider settings
        """
        self.config = config
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the provider-specific client.

        This method should set up any necessary client objects or
        connections needed to make API calls.
        """
        pass

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Make a call to the LLM provider.

        Args:
            prompt: The input prompt for the LLM
            **kwargs: Additional parameters:
                - model: Override the default model
                - max_tokens: Override max tokens
                - temperature: Override temperature
                - top_p: Top-p sampling parameter
                - stop: Stop sequences
                - stream: Enable streaming (if supported)

        Returns:
            LLMResponse with the generated content or error
        """
        pass

    def get_name(self) -> str:
        """Get the provider name."""
        return self.config.provider

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.config.model

    def update_api_key(self, api_key: str) -> None:
        """
        Update the API key for this provider.

        This is useful for key rotation.

        Args:
            api_key: New API key to use
        """
        self.config.api_key = api_key
        self._initialize_client()
