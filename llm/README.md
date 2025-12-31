# LLM Module Documentation

## Overview

The LLM module provides a unified, model-agnostic interface for calling Large Language Models across multiple providers (OpenAI, Anthropic, Groq, Ollama). It features intelligent routing, automatic failover, health tracking, and usage monitoring.

## Architecture

### Component Structure

```
LLMService
├── LLMRoutingConfig      # Tier-based routing configuration
├── Provider Implementations
│   ├── OpenAIProvider
│   ├── AnthropicProvider
│   ├── GroqProvider
│   └── OllamaProvider
└── Usage Tracking         # Automatic logging to persistence
```

### Key Features

1. **Model-Agnostic API** - Same interface for all providers
2. **Tier-Based Routing** - Automatic model selection based on task complexity
3. **Automatic Failover** - Retry with different models on rate limits
4. **Health Tracking** - Monitor key health and rate limits
5. **Round-Robin Key Rotation** - Distribute load across multiple API keys
6. **Usage Tracking** - Log all calls for observability

## Using LLM Service Standalone

You can use the LLM service independently without the full agent framework, taking advantage of routing, multi-provider support, and failover.

### Quick Start

```python
from curio_agent_sdk import initialize_llm_service, call_llm

# Initialize with default config from environment
initialize_llm_service()

# Simple call
response = call_llm("What is the capital of France?")
print(response)
```

### Configuration

The LLM service can be configured via environment variables or programmatically.

#### Environment Variables

**Minimal Configuration (Auto-Detection):**
```bash
OPENAI_API_KEY=sk-your-key-here
```

The SDK will auto-configure models for each tier based on available providers.

**Explicit Tier Configuration:**
```bash
OPENAI_API_KEY=sk-your-key-here

TIER1_MODELS=openai:gpt-4o-mini
TIER2_MODELS=openai:gpt-4o
TIER3_MODELS=openai:gpt-4o
```

**Multi-Provider with Failover:**
```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# If Groq fails, try OpenAI
TIER1_MODELS=groq:llama-3.1-8b-instant,openai:gpt-4o-mini
TIER2_MODELS=groq:llama-3.3-70b-versatile,openai:gpt-4o
TIER3_MODELS=openai:gpt-4o,groq:llama-3.3-70b-versatile
```

**Local Ollama:**
```bash
OLLAMA_HOST=http://localhost:11434

TIER1_MODELS=ollama:llama3.1:8b
TIER2_MODELS=ollama:llama3.1:70b
TIER3_MODELS=ollama:llama3.1:70b
```

#### Programmatic Configuration

```python
from curio_agent_sdk.llm import LLMService, LLMRoutingConfig

# Create routing config
routing = LLMRoutingConfig()

# Register custom provider
routing.register_custom_provider(
    provider_name="onprem-llm",
    api_key="sk-custom-key",
    default_model="llama-2-70b",
    base_url="https://llm.company.com/v1",
)

# Create service
service = LLMService(routing_config=routing)

# Use service
response = service.call_llm("Hello, world!", tier="tier1")
```

### Usage Patterns

#### 1. Simple Call (Auto-Routing)

```python
from curio_agent_sdk import call_llm

# Uses default tier (tier2) and auto-selects model
response = call_llm("Explain quantum computing")
print(response)
```

#### 2. Tier-Based Call

```python
from curio_agent_sdk import call_llm

# Use tier1 for fast/cheap tasks
response = call_llm("Summarize this text...", tier="tier1")

# Use tier3 for high-quality output
response = call_llm("Write a comprehensive essay...", tier="tier3")
```

#### 3. Explicit Provider/Model

```python
from curio_agent_sdk import call_llm

# Specify exact provider and model
response = call_llm(
    "Translate to French",
    provider="openai",
    model="gpt-4o"
)
```

#### 4. Detailed Response

```python
from curio_agent_sdk import call_llm_detailed

response = call_llm_detailed("Generate code", tier="tier2")

print(f"Content: {response.content}")
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Input tokens: {response.get_input_tokens()}")
print(f"Output tokens: {response.get_output_tokens()}")
print(f"Latency: {response.latency_ms}ms")
```

#### 5. With Custom Parameters

```python
from curio_agent_sdk import call_llm

response = call_llm(
    "Generate creative story",
    tier="tier3",
    temperature=0.9,
    max_tokens=2000,
    top_p=0.95,
)
```

#### 6. With Usage Tracking

```python
from curio_agent_sdk import initialize_llm_service, call_llm
from curio_agent_sdk.persistence import SQLitePersistence

# Initialize with persistence for tracking
persistence = SQLitePersistence("./llm_usage.db")
persistence.initialize_schema()

initialize_llm_service(persistence=persistence)

# All calls are automatically tracked
response = call_llm("Hello", tier="tier1", agent_id="my-agent", run_id="run-1")

# Query usage later
usage = persistence.get_llm_usage(agent_id="my-agent")
total_tokens = sum(u.get_total_tokens() or 0 for u in usage)
```

## Tier System

The SDK uses a three-tier system for different task complexities:

| Tier | Purpose | Default Models | Use Case |
|------|---------|----------------|----------|
| tier1 | Fast, simple tasks | llama-3.1-8b-instant | Simple Q&A, summarization |
| tier2 | Balanced quality/speed | llama-3.3-70b-versatile | General tasks, code generation |
| tier3 | High-quality output | gpt-4o | Complex reasoning, creative writing |

### Model Priority and Failover

The model list order in `TIER*_MODELS` defines priority. The first model is tried first; if it fails or hits rate limits, the next model is tried.

**Example:**
```bash
TIER1_MODELS=groq:llama-3.1-8b-instant,openai:gpt-4o-mini,anthropic:claude-3-haiku
```

This means:
1. Try Groq first
2. If Groq fails/rate-limited → try OpenAI
3. If OpenAI fails → try Anthropic

You can mix providers in any order.

## Routing Configuration

### LLMRoutingConfig

The `LLMRoutingConfig` manages tier-based routing with model priority.

**Key Concepts:**

1. **Model Priority List** - Ordered list of models to try (first is highest priority)
2. **Provider Configuration** - API keys, models, base URLs per provider
3. **Key Health Tracking** - Monitor success/failure rates per API key
4. **Round-Robin Key Rotation** - Distribute load across multiple keys

### Configuration Structure

```python
routing = LLMRoutingConfig()

# Providers are auto-loaded from environment
# Or register custom providers:
routing.register_custom_provider(
    provider_name="custom-llm",
    api_key="sk-key",
    default_model="model-name",
    base_url="https://api.example.com",
)

# Get provider/model for a tier
provider, model, key = routing.get_provider_and_model_for_tier("tier1")
```

### Multi-Key Support

Configure multiple API keys for load balancing and rate limit handling:

```bash
GROQ_API_KEY_1=key1
GROQ_API_KEY_1_NAME=primary
GROQ_API_KEY_2=key2
GROQ_API_KEY_2_NAME=secondary
GROQ_API_KEY_3=key3
```

The SDK automatically:
- Rotates through healthy keys (round-robin)
- Tracks health per key
- Skips rate-limited keys
- Recovers keys after timeout

### Health Tracking

The routing system tracks:
- Success/failure counts per key
- Rate limit detection
- Consecutive failures
- Automatic recovery after timeouts

```python
# Get routing statistics
service = get_llm_service()
stats = service.get_routing_stats()

print(stats)
# {
#   "providers": {
#     "groq": {
#       "enabled": True,
#       "key_count": 3,
#       "keys": {
#         "primary": {
#           "success_count": 100,
#           "failure_count": 2,
#           "rate_limit_hit": False,
#           "consecutive_failures": 0
#         }
#       }
#     }
#   },
#   "tiers": {...}
# }
```

## Provider Implementations

### Base Provider Interface

All providers inherit from `LLMProvider`:

```python
class LLMProvider(ABC):
    def __init__(self, config: LLMConfig)
    def _initialize_client(self) -> None  # Abstract
    def call(self, prompt: str, **kwargs) -> LLMResponse  # Abstract
```

### Supported Providers

#### OpenAI

**Configuration:**
```bash
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: custom endpoint
```

**Usage:**
```python
response = call_llm("Hello", provider="openai", model="gpt-4o")
```

**Models:**
- `gpt-4o` - Latest GPT-4
- `gpt-4o-mini` - Fast, cheaper variant
- `gpt-4-turbo` - Previous generation
- `gpt-3.5-turbo` - Legacy

#### Anthropic

**Configuration:**
```bash
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_DEFAULT_MODEL=claude-3-haiku-20240307
```

**Usage:**
```python
response = call_llm("Hello", provider="anthropic", model="claude-3-5-sonnet-20241022")
```

**Models:**
- `claude-3-5-sonnet-20241022` - Latest Claude
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fast, cheap

#### Groq

**Configuration:**
```bash
GROQ_API_KEY=gsk_...
GROQ_DEFAULT_MODEL=llama-3.1-8b-instant
```

**Usage:**
```python
response = call_llm("Hello", provider="groq", model="llama-3.3-70b-versatile")
```

**Models:**
- `llama-3.3-70b-versatile` - Latest, most capable
- `llama-3.1-70b-versatile` - Previous generation
- `llama-3.1-8b-instant` - Fast, cheap

#### Ollama (Local)

**Configuration:**
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.1:8b
```

**Usage:**
```python
response = call_llm("Hello", provider="ollama", model="llama3.1:70b")
```

**Models:**
- Any model available in your Ollama installation
- Format: `model:tag` (e.g., `llama3.1:8b`)

### Custom Providers

You can create custom providers for on-premise LLMs or custom inference endpoints:

```python
from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse

class CustomProvider(LLMProvider):
    def _initialize_client(self):
        # Initialize your client
        self.client = YourClient(self.config.api_key, self.config.base_url)
    
    def call(self, prompt: str, **kwargs) -> LLMResponse:
        model = kwargs.get("model", self.config.model)
        response = self.client.generate(prompt, model=model)
        
        return LLMResponse(
            content=response.text,
            provider="custom",
            model=model,
            usage=response.usage,
        )

# Register with service
from curio_agent_sdk.llm import LLMService, LLMConfig

config = LLMConfig(
    provider="custom",
    api_key="sk-key",
    base_url="https://api.example.com",
)

service = LLMService(custom_providers={"custom": CustomProvider})
service.register_provider("custom", CustomProvider, config)
```

## Response Models

### LLMResponse

Standardized response from all providers:

```python
@dataclass
class LLMResponse:
    content: str                    # Generated text
    provider: str                   # Provider name
    model: str                      # Model used
    usage: Optional[Dict[str, Any]] # Token usage (provider-specific)
    error: Optional[str]            # Error message if failed
    latency_ms: Optional[int]       # Response latency
```

**Methods:**
- `is_error()` - Check if response has error
- `is_rate_limited()` - Check if rate limited
- `get_input_tokens()` - Get input token count
- `get_output_tokens()` - Get output token count
- `get_total_tokens()` - Get total token count

### LLMConfig

Configuration for a provider:

```python
@dataclass
class LLMConfig:
    provider: str
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    base_url: Optional[str] = None
```

## Automatic Failover

The SDK automatically handles rate limits and failures:

1. **Rate Limit Detection** - Detects 429 errors and rate limit messages
2. **Model Exclusion** - Excludes rate-limited models from retry
3. **Next Model Selection** - Tries next model in priority list
4. **Key Rotation** - Rotates to next healthy API key
5. **Recovery** - Recovers keys after timeout (1 hour for rate limits, 5 minutes for failures)

**Example:**
```python
# If tier1 model is rate-limited, automatically tries next model
response = call_llm("Hello", tier="tier1")
# Tries: groq:llama-3.1-8b-instant → openai:gpt-4o-mini → anthropic:claude-3-haiku
```

## Usage Tracking

All LLM calls are automatically logged to persistence (if configured):

```python
from curio_agent_sdk import initialize_llm_service, call_llm
from curio_agent_sdk.persistence import SQLitePersistence

persistence = SQLitePersistence("./usage.db")
persistence.initialize_schema()

initialize_llm_service(persistence=persistence)

# Call is automatically tracked
call_llm("Hello", agent_id="agent-1", run_id="run-1")

# Query usage
usage = persistence.get_llm_usage(agent_id="agent-1")
for record in usage:
    print(f"{record.provider}/{record.model}: {record.get_total_tokens()} tokens")
```

## Best Practices

1. **Use Appropriate Tiers** - tier1 for simple tasks, tier3 for complex ones
2. **Configure Multiple Keys** - Avoid rate limits with key rotation
3. **Enable Usage Tracking** - Monitor costs and performance
4. **Handle Errors** - Always check `response.error` and `response.is_error()`
5. **Use Failover** - Let the SDK handle rate limits automatically
6. **Monitor Health** - Check routing stats regularly
7. **Custom Providers** - Use for on-premise or custom endpoints

## API Reference

### LLMService

**Methods:**
- `call_llm(prompt, provider, model, tier, **kwargs)` - Call LLM with routing
- `get_available_providers()` - List available providers
- `get_routing_stats()` - Get routing statistics
- `register_provider(name, class, config)` - Register custom provider

### Convenience Functions

- `call_llm(prompt, ...)` - Simple string response
- `call_llm_detailed(prompt, ...)` - Full LLMResponse object
- `initialize_llm_service(config, persistence, routing_config)` - Initialize global service
- `get_llm_service()` - Get global service instance

### LLMRoutingConfig

**Methods:**
- `get_provider_and_model_for_tier(tier, excluded_models)` - Get provider/model for tier
- `get_next_healthy_key(provider)` - Get next healthy API key
- `record_success(provider, key_name)` - Record successful call
- `record_failure(provider, key_name, is_rate_limit)` - Record failed call
- `get_stats()` - Get routing statistics
- `register_custom_provider(...)` - Register custom provider

See the source files for complete API documentation:
- `llm/service.py` - LLMService
- `llm/routing.py` - LLMRoutingConfig
- `llm/models.py` - Data models
- `llm/providers/base.py` - Provider interface

