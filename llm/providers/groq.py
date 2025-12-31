"""
Groq provider implementation with hybrid HTTP/client library approach.
"""

import logging
import requests
from typing import Optional

from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq library not installed. Install with: pip install groq")


class GroqProvider(LLMProvider):
    """
    Groq provider implementation with hybrid approach.

    Uses direct HTTP for immediate 429 detection (better for model rotation)
    and client library as fallback with built-in retries.

    Supports Llama, Mixtral, and other models via Groq's fast inference.

    Example:
        >>> config = LLMConfig(
        ...     provider="groq",
        ...     api_key="gsk_...",
        ...     model="llama-3.1-8b-instant",
        ... )
        >>> provider = GroqProvider(config)
        >>> response = provider.call("Hello, world!")
        >>> print(response.content)
    """

    GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"

    def _initialize_client(self) -> None:
        """Initialize Groq provider with both HTTP and client options."""
        if not self.config.api_key:
            raise ValueError("Groq API key not provided")

        # Initialize client library as fallback (with retries enabled)
        self.client_available = False
        if GROQ_AVAILABLE:
            try:
                self.client = groq.Groq(
                    api_key=self.config.api_key,
                    max_retries=2,
                )
                self.client_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client library: {e}")

    def call(self, prompt: str, use_client_library: bool = False, **kwargs) -> LLMResponse:
        """
        Call Groq API using either direct HTTP or client library.

        Args:
            prompt: The input prompt
            use_client_library: If True, use Groq client library (with retries).
                               If False, use direct HTTP (no retries, for rotation).
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        # Use client library if requested and available
        if use_client_library and self.client_available:
            return self._call_with_client_library(prompt, **kwargs)
        else:
            # Use direct HTTP for immediate 429 detection and model rotation
            return self._call_with_http(prompt, **kwargs)

    def _call_with_client_library(self, prompt: str, **kwargs) -> LLMResponse:
        """Call Groq API using client library (with automatic retries)."""
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
            if "stream" in kwargs:
                params["stream"] = kwargs["stream"]
            if "stop" in kwargs:
                params["stop"] = kwargs["stop"]

            logger.debug(f"Using Groq client library (with retries) for model: {model}")
            response = self.client.chat.completions.create(**params)

            return LLMResponse(
                content=response.choices[0].message.content,
                provider="groq",
                model=response.model,
                usage=response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None,
            )

        except Exception as e:
            return self._handle_error(e, model)

    def _call_with_http(self, prompt: str, **kwargs) -> LLMResponse:
        """Call Groq API using direct HTTP POST (no automatic retries)."""
        model = kwargs.get("model", self.config.model)

        # Prepare request payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stream" in kwargs:
            payload["stream"] = kwargs["stream"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.GROQ_API_BASE_URL}/chat/completions"

        try:
            logger.debug(f"Making HTTP POST to Groq API: model={model}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            status_code = response.status_code

            # Handle 429 rate limit error
            if status_code == 429:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    error_data = {"error": {"message": response.text or "Rate limit exceeded"}}

                error_message = error_data.get("error", {}).get("message", "Rate limit exceeded (429)")
                logger.warning(f"Groq API 429 rate limit: {error_message}")

                return LLMResponse(
                    content="",
                    provider="groq",
                    model=model,
                    error=f"Rate limit exceeded: {error_message}",
                    usage={"is_rate_limit": True, "status_code": 429, "error_type": "RateLimitError"},
                )

            # Handle other HTTP errors
            if status_code >= 400:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    error_data = {"error": {"message": response.text or f"HTTP {status_code} error"}}

                error_message = error_data.get("error", {}).get("message", f"HTTP {status_code} error")
                logger.error(f"Groq API error ({status_code}): {error_message}")

                return LLMResponse(
                    content="",
                    provider="groq",
                    model=model,
                    error=f"HTTP {status_code}: {error_message}",
                    usage={"is_rate_limit": False, "status_code": status_code},
                )

            # Parse successful response
            response_data = response.json()

            if "choices" not in response_data or not response_data["choices"]:
                error_msg = "Invalid response format from Groq API"
                logger.error(error_msg)
                return LLMResponse(
                    content="",
                    provider="groq",
                    model=model,
                    error=error_msg,
                )

            choice = response_data["choices"][0]
            content = choice.get("message", {}).get("content", "")
            response_model = response_data.get("model", model)

            # Extract usage information
            usage_info = response_data.get("usage")

            logger.debug(
                f"Groq API success: model={response_model}, "
                f"tokens={usage_info.get('total_tokens', 'unknown') if usage_info else 'unknown'}"
            )

            return LLMResponse(
                content=content,
                provider="groq",
                model=response_model,
                usage=usage_info,
            )

        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            logger.error(f"Groq API timeout: {error_msg}")
            return LLMResponse(
                content="",
                provider="groq",
                model=model,
                error=error_msg,
            )

        except requests.exceptions.RequestException as e:
            error_str = str(e)
            error_type = type(e).__name__
            logger.error(f"Groq API request exception ({error_type}): {error_str}")

            # Check if it's a connection error that might indicate rate limiting
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

            return LLMResponse(
                content="",
                provider="groq",
                model=model,
                error=error_str,
                usage={"is_rate_limit": is_rate_limit, "error_type": error_type} if is_rate_limit else None,
            )

        except Exception as e:
            return self._handle_error(e, model)

    def _handle_error(self, e: Exception, model: str) -> LLMResponse:
        """Handle an exception and return appropriate LLMResponse."""
        error_str = str(e)
        error_type = type(e).__name__

        # Check for rate limit errors
        is_rate_limit = False
        status_code = None

        # Check exception type
        if any(keyword in error_type.lower() for keyword in ["ratelimit", "rate_limit", "throttle"]):
            is_rate_limit = True
            logger.info(f"Detected rate limit from exception type: {error_type}")

        # Check status code
        if hasattr(e, 'status_code'):
            status_code = e.status_code
            if status_code == 429:
                is_rate_limit = True
        elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            status_code = e.response.status_code
            if status_code == 429:
                is_rate_limit = True

        # Check error message
        if not is_rate_limit:
            error_lower = error_str.lower()
            if any(keyword in error_lower for keyword in ["rate limit", "429", "too many requests", "quota exceeded", "throttled"]):
                is_rate_limit = True

        logger.error(f"Groq error ({error_type}): {error_str} (rate_limit: {is_rate_limit})")

        return LLMResponse(
            content="",
            provider="groq",
            model=model,
            error=error_str,
            usage={"is_rate_limit": is_rate_limit, "status_code": status_code, "error_type": error_type} if is_rate_limit else None,
        )
