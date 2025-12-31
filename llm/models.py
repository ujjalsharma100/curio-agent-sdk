"""
Data models for the LLM service.

This module contains configuration and response models used throughout
the LLM service layer.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class LLMConfig:
    """
    Configuration for an LLM provider.

    Attributes:
        provider: Provider name (openai, anthropic, groq, ollama)
        api_key: API key for authentication (not needed for ollama)
        model: Default model to use
        max_tokens: Default maximum tokens for responses
        temperature: Default temperature for generation
        base_url: Base URL for API calls (used for custom endpoints)
    """
    provider: str
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    base_url: Optional[str] = None

    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.provider:
            return False
        if self.provider in ("openai", "anthropic", "groq") and not self.api_key:
            return False
        return True


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.

    This class normalizes responses from different providers into
    a consistent format.

    Attributes:
        content: The generated text content
        provider: Name of the provider that handled the request
        model: Name of the model used
        usage: Token usage information (provider-specific format)
        error: Error message if the request failed
        latency_ms: Response latency in milliseconds
    """
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None

    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return bool(self.error)

    def is_rate_limited(self) -> bool:
        """Check if this response indicates a rate limit error."""
        if not self.error:
            return False

        # Check usage dict for rate limit flag
        if self.usage and isinstance(self.usage, dict):
            if self.usage.get("is_rate_limit"):
                return True

        # Check error message for rate limit indicators
        error_lower = self.error.lower()
        rate_limit_keywords = [
            "rate limit",
            "429",
            "too many requests",
            "quota exceeded",
            "throttled",
            "rate_limit",
        ]
        return any(keyword in error_lower for keyword in rate_limit_keywords)

    def get_input_tokens(self) -> Optional[int]:
        """Get input token count if available."""
        if not self.usage:
            return None
        return (
            self.usage.get("prompt_tokens") or
            self.usage.get("input_tokens")
        )

    def get_output_tokens(self) -> Optional[int]:
        """Get output token count if available."""
        if not self.usage:
            return None
        return (
            self.usage.get("completion_tokens") or
            self.usage.get("output_tokens")
        )

    def get_total_tokens(self) -> Optional[int]:
        """Get total token count if available."""
        if not self.usage:
            return None

        # Try direct total first
        total = self.usage.get("total_tokens")
        if total:
            return total

        # Calculate from input + output
        input_t = self.get_input_tokens()
        output_t = self.get_output_tokens()
        if input_t is not None and output_t is not None:
            return input_t + output_t

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }
