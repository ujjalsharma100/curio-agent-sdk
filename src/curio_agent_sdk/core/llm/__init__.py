"""
LLM module for Curio Agent SDK.

Provides async, message-based, provider-native LLM interface with:
- Native tool/function calling
- Streaming responses
- Tiered model routing with automatic failover
- Round-robin key rotation with health tracking
- Per-request API keys (thread-safe)
"""

from curio_agent_sdk.core.llm.client import LLMClient
from curio_agent_sdk.core.llm.batch_client import BatchLLMClient
from curio_agent_sdk.core.llm.router import TieredRouter, RouteResult, ProviderConfig, ProviderKey
from curio_agent_sdk.core.llm.providers.base import LLMProvider
from curio_agent_sdk.core.llm.providers.openai import OpenAIProvider
from curio_agent_sdk.core.llm.providers.anthropic import AnthropicProvider
from curio_agent_sdk.core.llm.providers.groq import GroqProvider
from curio_agent_sdk.core.llm.providers.ollama import OllamaProvider
from curio_agent_sdk.core.llm.token_counter import count_tokens

__all__ = [
    "LLMClient",
    "BatchLLMClient",
    "TieredRouter",
    "RouteResult",
    "ProviderConfig",
    "ProviderKey",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GroqProvider",
    "OllamaProvider",
    "count_tokens",
]
