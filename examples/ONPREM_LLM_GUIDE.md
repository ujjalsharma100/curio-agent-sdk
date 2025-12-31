# On-Premise LLM Configuration Guide

This guide explains how to configure the Curio Agent SDK to use your own on-premise deployed LLM models.

## Overview

The SDK supports custom LLM providers, allowing you to:
- Use your own on-premise LLM infrastructure
- Configure custom inference endpoints
- Integrate with company-specific LLM deployments
- Mix on-prem and cloud models in your routing configuration

## Two Approaches

### Approach 1: OpenAI-Compatible API (Recommended)

If your on-prem LLM server uses an OpenAI-compatible API format (most modern inference servers do), you can simply use the `OpenAIProvider` with a custom `base_url`.

**Advantages:**
- Simple configuration
- No custom code needed
- Works with vLLM, TGI, and other OpenAI-compatible servers

**Example:**

```python
from curio_agent_sdk.llm.routing import LLMRoutingConfig
from curio_agent_sdk.llm.service import LLMService
from curio_agent_sdk.llm.providers.openai import OpenAIProvider
from curio_agent_sdk import InMemoryPersistence

# Create routing config
routing_config = LLMRoutingConfig()

# Register your on-prem provider
routing_config.register_custom_provider(
    provider_name="onprem-llm",
    api_key="sk-your-api-key",  # Your API key (or "" if not required)
    default_model="llama-2-70b-chat",  # Your model name
    base_url="https://llm.yourcompany.com/v1",  # Your endpoint
)

# Configure tiers
custom_tiers = {
    "tier1": ["onprem-llm:llama-2-70b-chat"],
    "tier2": ["onprem-llm:llama-2-70b-chat"],
    "tier3": ["onprem-llm:llama-2-70b-chat"],
}

routing_config = LLMRoutingConfig(custom_tiers=custom_tiers)
routing_config.register_custom_provider(
    provider_name="onprem-llm",
    api_key="sk-your-api-key",
    default_model="llama-2-70b-chat",
    base_url="https://llm.yourcompany.com/v1",
)

# Create LLM service
llm_service = LLMService(
    persistence=InMemoryPersistence(),
    routing_config=routing_config,
    custom_providers={"onprem-llm": OpenAIProvider},  # Use OpenAIProvider
)

# Use it
response = llm_service.call_llm(
    prompt="Hello!",
    provider="onprem-llm",
    model="llama-2-70b-chat",
)
```

### Approach 2: Custom Provider Class

If your on-prem LLM has a non-standard API format, create a custom provider class.

**Example:**

```python
from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse
import requests

class CustomOnPremProvider(LLMProvider):
    """Custom provider for non-standard API format."""
    
    def _initialize_client(self):
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
            })
    
    def call(self, prompt: str, **kwargs) -> LLMResponse:
        model = kwargs.get("model", self.config.model)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        response = self.session.post(url, json=payload)
        data = response.json()
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            provider="custom-onprem",
            model=model,
        )

# Register and use
routing_config.register_custom_provider(
    provider_name="custom-onprem",
    api_key="sk-key",
    default_model="company-llm-v1",
    base_url="https://llm.company.com/api/v1",
)

llm_service = LLMService(
    routing_config=routing_config,
    custom_providers={"custom-onprem": CustomOnPremProvider},
)
```

## Using with Agents

You can configure agents to use on-prem LLMs:

```python
from curio_agent_sdk import BaseAgent, InMemoryPersistence

class MyAgent(BaseAgent):
    def __init__(self, agent_id: str, onprem_config: dict):
        # Create LLM service with on-prem config
        llm_service = create_onprem_llm_service(onprem_config)
        
        super().__init__(
            agent_id=agent_id,
            persistence=InMemoryPersistence(),
            llm_service=llm_service,
        )
    
    # ... rest of agent implementation

# Use it
agent = MyAgent(
    agent_id="my-agent",
    onprem_config={
        "base_url": "https://llm.company.com/v1",
        "api_key": "sk-key",
        "model": "llama-2-70b-chat",
    }
)
```

## Environment Variables

You can also configure on-prem providers via environment variables by extending the routing configuration. However, for custom providers, programmatic configuration is recommended.

## Complete Examples

See the following example files:
- `custom_onprem_provider.py` - Shows both approaches with detailed examples
- `onprem_llm_agent_example.py` - Complete agent example using on-prem LLM

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify the `base_url` is correct and accessible
   - Check network/firewall settings
   - Ensure the endpoint is running

2. **Authentication Errors**
   - Verify the `api_key` is correct
   - Check if your endpoint requires authentication
   - Some endpoints use empty string for no-auth

3. **Model Not Found**
   - Verify the `model` name matches your deployed model
   - Check model availability on your endpoint

4. **API Format Mismatch**
   - If using Approach 1, ensure your endpoint is OpenAI-compatible
   - If not, use Approach 2 with a custom provider class
   - Check your endpoint's API documentation

### Testing

Test your configuration:

```python
# Test the provider directly
response = llm_service.call_llm(
    prompt="Hello, world!",
    provider="onprem-llm",
    model="your-model-name",
)

if response.error:
    print(f"Error: {response.error}")
else:
    print(f"Success: {response.content}")
```

## Best Practices

1. **Use OpenAI-Compatible Format**: If possible, deploy your on-prem LLM with OpenAI-compatible API format for easier integration.

2. **Error Handling**: Implement proper error handling in custom providers.

3. **Rate Limiting**: Consider rate limiting and retry logic in custom providers.

4. **Monitoring**: Track usage and performance of on-prem providers.

5. **Fallback**: Consider mixing on-prem and cloud models for redundancy.

## Support

For questions or issues:
- Check the example files in `examples/`
- Review the SDK documentation
- Open an issue on the repository

