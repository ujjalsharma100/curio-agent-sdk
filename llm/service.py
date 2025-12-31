"""
Main LLM Service that manages different providers.

This module provides the LLMService class which serves as the unified
interface for calling LLMs through different providers with automatic
routing, failover, and usage tracking.
"""

import logging
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from curio_agent_sdk.llm.models import LLMConfig, LLMResponse
from curio_agent_sdk.llm.routing import LLMRoutingConfig, ProviderKey
from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.providers.openai import OpenAIProvider
from curio_agent_sdk.llm.providers.anthropic import AnthropicProvider
from curio_agent_sdk.llm.providers.groq import GroqProvider
from curio_agent_sdk.llm.providers.ollama import OllamaProvider
from curio_agent_sdk.core.models import AgentLLMUsage

logger = logging.getLogger(__name__)


class LLMService:
    """
    Main LLM service that manages different providers.

    This class provides a unified interface for calling LLMs with features like:
    - Model-agnostic API (call any provider with same interface)
    - Tiered routing (automatically select provider/model based on task complexity)
    - Automatic failover (retry with different models on rate limits)
    - Usage tracking (log all calls for observability)
    - Key rotation (round-robin through multiple API keys)

    Example:
        >>> # Initialize with default config from environment
        >>> service = LLMService()
        >>>
        >>> # Simple call
        >>> response = service.call_llm("Hello, world!")
        >>>
        >>> # Tier-based call (automatic provider/model selection)
        >>> response = service.call_llm(
        ...     "Write a poem about AI",
        ...     tier="tier3",  # Use high-quality model
        ... )
        >>>
        >>> # Explicit provider/model
        >>> response = service.call_llm(
        ...     "Summarize this text...",
        ...     provider="openai",
        ...     model="gpt-4",
        ... )
    """

    # Maximum retries for rate limit handling
    MAX_RETRIES = 10

    def __init__(
        self,
        config: Optional[Any] = None,
        persistence: Optional[Any] = None,
        routing_config: Optional[LLMRoutingConfig] = None,
        custom_providers: Optional[Dict[str, type]] = None,
    ):
        """
        Initialize the LLM service.

        Args:
            config: Optional AgentConfig for provider configuration
            persistence: Optional BasePersistence for usage tracking
            routing_config: Optional custom LLMRoutingConfig
            custom_providers: Optional dict of provider_name -> ProviderClass for custom providers
        """
        self.config = config
        self.persistence = persistence
        self.providers: Dict[str, LLMProvider] = {}
        self.routing_config = routing_config or LLMRoutingConfig()
        self.custom_providers = custom_providers or {}

        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers."""
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
            "groq": GroqProvider,
        }
        
        # Merge custom providers (custom providers take precedence)
        provider_classes.update(self.custom_providers)

        for provider_name, provider_class in provider_classes.items():
            try:
                # Get config from routing config
                if provider_name in self.routing_config.providers:
                    routing_provider = self.routing_config.providers[provider_name]
                    if routing_provider.keys:
                        key = routing_provider.keys[0]
                        llm_config = LLMConfig(
                            provider=provider_name,
                            api_key=key.api_key,
                            model=routing_provider.default_model,
                            base_url=routing_provider.base_url,
                        )
                        self.providers[provider_name] = provider_class(llm_config)
                        logger.info(f"Initialized {provider_name} provider")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name} provider: {str(e)}")
    
    def register_provider(self, provider_name: str, provider_class: type, provider_config: LLMConfig):
        """
        Register a custom provider at runtime.
        
        Args:
            provider_name: Name of the provider
            provider_class: Provider class (must inherit from LLMProvider)
            provider_config: LLMConfig for the provider
        """
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(f"Provider class must inherit from LLMProvider")
        
        self.providers[provider_name] = provider_class(provider_config)
        logger.info(f"Registered custom provider: {provider_name}")

    def call_llm(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        excluded_models: Optional[List[str]] = None,
        retry_count: int = 0,
        **kwargs,
    ) -> LLMResponse:
        """
        Call the LLM with automatic routing and failover.

        Args:
            prompt: The input prompt for the LLM
            provider: Specific provider to use (optional)
            model: Specific model to use (optional)
            tier: Tier for automatic routing ("tier1", "tier2", "tier3")
            run_id: Optional run ID for tracking
            agent_id: Optional agent ID for tracking
            excluded_models: List of models to exclude (for retries)
            retry_count: Internal retry counter
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            LLMResponse with generated content or error
        """
        if excluded_models is None:
            excluded_models = []

        # Safety check: prevent infinite recursion
        if retry_count >= self.MAX_RETRIES:
            error_msg = f"Max retries ({self.MAX_RETRIES}) reached for tier {tier}. Excluded models: {excluded_models}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                provider="unknown",
                model="unknown",
                error=error_msg,
            )

        selected_provider = provider
        selected_model = model
        selected_key: Optional[ProviderKey] = None
        key_name = None
        use_client_library = False

        # If tier specified, get provider, model, and key from routing
        if tier:
            selected_provider, selected_model, selected_key = \
                self.routing_config.get_provider_and_model_for_tier(tier, excluded_models=excluded_models)

            # If no model available and we have excluded models, try client library fallback
            if not selected_provider and excluded_models:
                tier_config = self.routing_config.tiers.get(tier)
                if tier_config and tier_config.model_priority:
                    # Find first groq model in priority list for client library fallback
                    for mp in tier_config.model_priority:
                        if mp.provider == "groq":
                            selected_provider = "groq"
                            selected_model = mp.model
                            selected_key = self.routing_config.get_next_healthy_key("groq")
                            use_client_library = True
                            logger.info(
                                f"All models exhausted in tier {tier}. "
                                f"Using client library with retries for {selected_model}"
                            )
                            break

            if not selected_provider:
                error_msg = f"No available provider/model for tier {tier} (excluded models: {excluded_models})"
                logger.error(error_msg)
                return LLMResponse(
                    content="",
                    provider="unknown",
                    model="unknown",
                    error=error_msg,
                )

            key_name = selected_key.name or "default" if selected_key else "default"
            logger.info(
                f"Tier {tier} routing: {selected_provider}/{selected_model} "
                f"with key {key_name} (excluded: {excluded_models}, retry: {retry_count})"
            )
        else:
            # Explicit provider specified, get key
            if selected_provider:
                selected_provider = selected_provider.lower()
                selected_key = self.routing_config.get_next_healthy_key(selected_provider)
                if not selected_key:
                    error_msg = f"No healthy keys available for {selected_provider}"
                    logger.error(error_msg)
                    return LLMResponse(
                        content="",
                        provider=selected_provider,
                        model=model or "unknown",
                        error=error_msg,
                    )
                key_name = selected_key.name or "default"
            else:
                # Use first available provider
                if self.providers:
                    selected_provider = list(self.providers.keys())[0]
                else:
                    return LLMResponse(
                        content="",
                        provider="unknown",
                        model="unknown",
                        error="No providers configured",
                    )

        # Check if provider is available
        if selected_provider not in self.providers:
            available_providers = list(self.providers.keys())
            error_msg = f"Provider '{selected_provider}' not available. Available: {available_providers}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                provider=selected_provider,
                model="unknown",
                error=error_msg,
            )

        # Get provider instance
        provider_instance = self.providers[selected_provider]
        original_api_key = provider_instance.config.api_key

        # Temporarily use selected key
        if selected_key:
            provider_instance.config.api_key = selected_key.api_key

        # Get model config from routing if available
        if selected_model and selected_provider in self.routing_config.providers:
            provider_routing_config = self.routing_config.providers[selected_provider]
            if selected_model in provider_routing_config.models:
                model_config = provider_routing_config.models[selected_model]
                if 'max_tokens' not in kwargs:
                    kwargs['max_tokens'] = model_config.max_tokens
                if 'temperature' not in kwargs:
                    kwargs['temperature'] = model_config.temperature

        # Use selected model
        if selected_model:
            kwargs['model'] = selected_model

        # Track timing
        start_time = time.time()

        # Make the call
        logger.info(
            f"Calling {selected_provider}/{selected_model or 'default'} "
            f"with prompt length: {len(prompt)}"
        )

        try:
            # Pass use_client_library flag to Groq provider
            if selected_provider == "groq" and use_client_library:
                kwargs['use_client_library'] = True

            response = provider_instance.call(prompt, **kwargs)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            response.latency_ms = latency_ms

            # Record success
            if key_name:
                self.routing_config.record_success(selected_provider, key_name)

            if response.error:
                logger.error(f"LLM call failed: {response.error}")

                # Check if it's a rate limit error
                if response.is_rate_limited():
                    # Record failure for key
                    if key_name:
                        self.routing_config.record_failure(selected_provider, key_name, is_rate_limit=True)

                    # If tier-based and rate-limited, try next model/key
                    if tier:
                        new_excluded_models = excluded_models.copy()
                        if selected_model and selected_model not in new_excluded_models:
                            new_excluded_models.append(selected_model)
                            logger.warning(
                                f"Rate limit hit for {selected_provider}/{selected_model}. "
                                f"Excluding model. Trying next..."
                            )

                        # Restore original key
                        provider_instance.config.api_key = original_api_key

                        return self.call_llm(
                            prompt,
                            tier=tier,
                            run_id=run_id,
                            agent_id=agent_id,
                            excluded_models=new_excluded_models,
                            retry_count=retry_count + 1,
                            **kwargs,
                        )
            else:
                logger.info(f"LLM call successful. Response length: {len(response.content)}")

            # Track LLM usage
            model_name = response.model or selected_model or provider_instance.config.model
            input_params = {
                "model": model_name,
                "temperature": kwargs.get("temperature", provider_instance.config.temperature),
                "max_tokens": kwargs.get("max_tokens", provider_instance.config.max_tokens),
            }

            self._track_llm_usage(
                agent_id=agent_id,
                run_id=run_id,
                provider=selected_provider,
                model=model_name,
                prompt=prompt,
                input_params=input_params,
                response=response,
            )

            return response

        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            logger.error(f"LLM call exception ({error_type}): {error_str}")

            # Create error response
            error_response = LLMResponse(
                content="",
                provider=selected_provider,
                model=selected_model or "unknown",
                error=error_str,
            )

            # Check if it's a rate limit error
            if error_response.is_rate_limited() or (
                hasattr(e, 'status_code') and e.status_code == 429
            ):
                # Record failure
                if key_name:
                    self.routing_config.record_failure(selected_provider, key_name, is_rate_limit=True)

                # If tier-based, try next model/key
                if tier:
                    new_excluded_models = excluded_models.copy()
                    if selected_model and selected_model not in new_excluded_models:
                        new_excluded_models.append(selected_model)

                    # Restore original key
                    provider_instance.config.api_key = original_api_key

                    return self.call_llm(
                        prompt,
                        tier=tier,
                        run_id=run_id,
                        agent_id=agent_id,
                        excluded_models=new_excluded_models,
                        retry_count=retry_count + 1,
                        **kwargs,
                    )

            # Restore original key
            provider_instance.config.api_key = original_api_key

            return error_response

        finally:
            # Always restore original API key
            provider_instance.config.api_key = original_api_key

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return self.routing_config.get_stats()

    def _track_llm_usage(
        self,
        agent_id: Optional[str],
        run_id: Optional[str],
        provider: str,
        model: str,
        prompt: str,
        input_params: Dict[str, Any],
        response: LLMResponse,
    ):
        """Track LLM usage in database."""
        if not self.persistence:
            return

        try:
            llm_usage = AgentLLMUsage(
                agent_id=agent_id,
                run_id=run_id,
                provider=provider,
                model=model,
                prompt=prompt,
                prompt_length=len(prompt),
                input_params=json.dumps(input_params),
                input_tokens=response.get_input_tokens(),
                output_tokens=response.get_output_tokens(),
                response_content=response.content,
                response_length=len(response.content) if response.content else None,
                usage_metrics=json.dumps(response.usage) if response.usage else None,
                status="error" if response.error else "success",
                error_message=response.error,
                latency_ms=response.latency_ms,
                created_at=datetime.now(),
            )

            self.persistence.log_llm_usage(llm_usage)
        except Exception as e:
            logger.error(f"Failed to track LLM usage: {e}")


# Global service instance
_llm_service: Optional[LLMService] = None


def initialize_llm_service(
    config: Optional[Any] = None,
    persistence: Optional[Any] = None,
    routing_config: Optional[LLMRoutingConfig] = None,
) -> LLMService:
    """
    Initialize the global LLM service instance.

    Args:
        config: Optional AgentConfig
        persistence: Optional BasePersistence
        routing_config: Optional LLMRoutingConfig

    Returns:
        The initialized LLMService
    """
    global _llm_service
    _llm_service = LLMService(
        config=config,
        persistence=persistence,
        routing_config=routing_config,
    )
    logger.info("LLM service initialized")
    return _llm_service


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def call_llm(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tier: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function for calling LLM with just a prompt.

    Args:
        prompt: The input prompt for the LLM
        provider: Specific provider to use (optional)
        model: Specific model to use (optional)
        tier: Tier for automatic routing
        run_id: Optional run ID for tracking
        agent_id: Optional agent ID for tracking
        **kwargs: Additional parameters

    Returns:
        str: The response content, or error message if failed
    """
    service = get_llm_service()
    response = service.call_llm(
        prompt,
        provider=provider,
        model=model,
        tier=tier,
        run_id=run_id or "default",
        agent_id=agent_id or "default",
        **kwargs,
    )

    if response.error:
        return f"Error: {response.error}"
    return response.content


def call_llm_detailed(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tier: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs,
) -> LLMResponse:
    """
    Detailed LLM caller that returns full response object.

    Args:
        prompt: The input prompt for the LLM
        provider: Specific provider to use (optional)
        model: Specific model to use (optional)
        tier: Tier for automatic routing
        run_id: Optional run ID for tracking
        agent_id: Optional agent ID for tracking
        **kwargs: Additional parameters

    Returns:
        LLMResponse: Full response object with metadata
    """
    service = get_llm_service()
    return service.call_llm(
        prompt,
        provider=provider,
        model=model,
        tier=tier,
        run_id=run_id,
        agent_id=agent_id,
        **kwargs,
    )
