"""
LLM module for Curio Agent SDK.

Provides async, message-based, provider-native LLM interface with:
- Native tool/function calling
- Streaming responses
- Tiered model routing with automatic failover
- Round-robin key rotation with health tracking
- Per-request API keys (thread-safe)
"""

from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.llm.batch_client import BatchLLMClient
from curio_agent_sdk.llm.router import TieredRouter, RouteResult, ProviderConfig, ProviderKey
from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.llm.providers.openai import OpenAIProvider
from curio_agent_sdk.llm.providers.anthropic import AnthropicProvider
from curio_agent_sdk.llm.providers.groq import GroqProvider
from curio_agent_sdk.llm.providers.ollama import OllamaProvider
from curio_agent_sdk.llm.token_counter import count_tokens

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
