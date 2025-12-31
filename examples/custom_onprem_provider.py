"""
Custom On-Prem LLM Provider Example

This example demonstrates how to create and configure a custom LLM provider
for on-premise deployed LLM models. This is useful when companies want to
use their own internal LLM infrastructure.

There are two approaches:
1. Use OpenAIProvider with custom base_url (for OpenAI-compatible APIs)
2. Create a fully custom provider (for non-standard APIs)

This example shows both approaches.
"""

import logging
from typing import Optional, Dict, Any

from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.providers.openai import OpenAIProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse
from curio_agent_sdk.llm.service import LLMService
from curio_agent_sdk.llm.routing import LLMRoutingConfig, ProviderKey, ProviderConfig
from curio_agent_sdk import AgentConfig, InMemoryPersistence

logger = logging.getLogger(__name__)


# ============================================================================
# APPROACH 1: Use OpenAIProvider with custom base_url
# ============================================================================
# This is the simplest approach if your on-prem LLM uses an OpenAI-compatible API.
# Most modern LLM inference servers (vLLM, TGI, etc.) support OpenAI-compatible endpoints.

def configure_onprem_with_openai_provider():
    """
    Configure on-prem LLM using OpenAIProvider with custom base_url.
    
    This works if your on-prem LLM server implements the OpenAI API format.
    """
    # Create routing config
    routing_config = LLMRoutingConfig()
    
    # Register your on-prem provider
    routing_config.register_custom_provider(
        provider_name="onprem-llm",
        api_key="sk-your-onprem-key",  # Your on-prem API key (if required)
        default_model="llama-2-70b-chat",  # Your model name
        base_url="https://llm.yourcompany.com/v1",  # Your on-prem endpoint
        key_name="onprem-key-1",
    )
    
    # Configure tiers to use your on-prem model
    custom_tiers = {
        "tier1": ["onprem-llm:llama-2-70b-chat"],
        "tier2": ["onprem-llm:llama-2-70b-chat"],
        "tier3": ["onprem-llm:llama-2-70b-chat"],
    }
    
    # Update routing config with custom tiers
    routing_config = LLMRoutingConfig(custom_tiers=custom_tiers)
    routing_config.register_custom_provider(
        provider_name="onprem-llm",
        api_key="sk-your-onprem-key",
        default_model="llama-2-70b-chat",
        base_url="https://llm.yourcompany.com/v1",
    )
    
    # Create LLM service with custom routing
    persistence = InMemoryPersistence()
    
    # Register OpenAIProvider as the provider class for "onprem-llm"
    # Since it uses OpenAI-compatible API, we can reuse OpenAIProvider
    llm_service = LLMService(
        persistence=persistence,
        routing_config=routing_config,
        custom_providers={"onprem-llm": OpenAIProvider},  # Use OpenAIProvider for OpenAI-compatible API
    )
    
    return llm_service


# ============================================================================
# APPROACH 2: Create a fully custom provider
# ============================================================================
# Use this if your on-prem LLM has a non-standard API format.

class CustomOnPremProvider(LLMProvider):
    """
    Custom provider for on-prem LLM with non-standard API.
    
    This example shows how to implement a provider for a custom API format.
    You'll need to adapt this to match your actual on-prem LLM API.
    """
    
    def _initialize_client(self) -> None:
        """Initialize client for your on-prem LLM."""
        # Example: Using requests library for HTTP calls
        import requests
        
        if not self.config.base_url:
            raise ValueError("base_url is required for custom on-prem provider")
        
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.session = requests.Session()
        
        # Set up authentication headers if needed
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
    
    def call(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Call your on-prem LLM API.
        
        This is a template - adapt the API call format to match your actual endpoint.
        """
        model = kwargs.get("model", self.config.model)
        
        try:
            # Example API call format (adapt to your actual API)
            url = f"{self.base_url}/chat/completions"
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
            
            # Add optional parameters
            if "top_p" in kwargs:
                payload["top_p"] = kwargs["top_p"]
            if "stop" in kwargs:
                payload["stop"] = kwargs["stop"]
            
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response (adapt to your API's response format)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            
            return LLMResponse(
                content=content,
                provider="custom-onprem",
                model=model,
                usage=usage,
            )
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Custom on-prem LLM API error: {error_str}")
            
            # Check for rate limit
            is_rate_limit = (
                "rate limit" in error_str.lower() or
                "429" in error_str or
                (hasattr(e, 'response') and e.response and e.response.status_code == 429)
            )
            
            return LLMResponse(
                content="",
                provider="custom-onprem",
                model=model,
                error=error_str,
                usage={"is_rate_limit": is_rate_limit} if is_rate_limit else None,
            )


def configure_onprem_with_custom_provider():
    """
    Configure on-prem LLM using a fully custom provider.
    
    Use this if your on-prem LLM has a non-standard API format.
    """
    # Create routing config
    routing_config = LLMRoutingConfig()
    
    # Register your on-prem provider
    routing_config.register_custom_provider(
        provider_name="custom-onprem",
        api_key="sk-your-custom-key",
        default_model="company-llm-v1",
        base_url="https://llm.yourcompany.com/api/v1",
    )
    
    # Configure tiers
    custom_tiers = {
        "tier1": ["custom-onprem:company-llm-v1"],
        "tier2": ["custom-onprem:company-llm-v1"],
        "tier3": ["custom-onprem:company-llm-v1"],
    }
    
    routing_config = LLMRoutingConfig(custom_tiers=custom_tiers)
    routing_config.register_custom_provider(
        provider_name="custom-onprem",
        api_key="sk-your-custom-key",
        default_model="company-llm-v1",
        base_url="https://llm.yourcompany.com/api/v1",
    )
    
    # Create LLM service with custom provider
    persistence = InMemoryPersistence()
    llm_service = LLMService(
        persistence=persistence,
        routing_config=routing_config,
        custom_providers={"custom-onprem": CustomOnPremProvider},
    )
    
    return llm_service


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of using the on-prem LLM provider."""
    
    print("=" * 60)
    print("On-Prem LLM Provider Example")
    print("=" * 60)
    
    # Choose your approach:
    # Option 1: OpenAI-compatible API (simpler)
    print("\n[Option 1] Using OpenAIProvider with custom base_url...")
    try:
        llm_service = configure_onprem_with_openai_provider()
        
        # Test the provider
        response = llm_service.call_llm(
            prompt="Hello! Can you tell me about AI?",
            provider="onprem-llm",
            model="llama-2-70b-chat",
        )
        
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(f"Response: {response.content[:200]}...")
            print(f"Provider: {response.provider}, Model: {response.model}")
    except Exception as e:
        print(f"Error (this is expected if endpoint is not available): {e}")
    
    # Option 2: Custom provider (for non-standard APIs)
    print("\n[Option 2] Using custom provider class...")
    try:
        llm_service = configure_onprem_with_custom_provider()
        
        # Test the provider
        response = llm_service.call_llm(
            prompt="Hello! Can you tell me about AI?",
            provider="custom-onprem",
            model="company-llm-v1",
        )
        
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(f"Response: {response.content[:200]}...")
            print(f"Provider: {response.provider}, Model: {response.model}")
    except Exception as e:
        print(f"Error (this is expected if endpoint is not available): {e}")
    
    print("\n" + "=" * 60)
    print("Configuration Complete!")
    print("=" * 60)
    print("\nTo use in your code:")
    print("1. Update the base_url to point to your on-prem endpoint")
    print("2. Update the api_key if your endpoint requires authentication")
    print("3. Update the model name to match your deployed model")
    print("4. If using custom provider, adapt the API call format in CustomOnPremProvider.call()")


if __name__ == "__main__":
    example_usage()

