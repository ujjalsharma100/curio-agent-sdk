"""
LLM module for Curio Agent SDK.

This module provides a unified interface for interacting with different
LLM providers (OpenAI, Anthropic, Groq, Ollama) with features like:
- Model-agnostic API
- Tiered model routing with automatic failover
- Health tracking and rate limit handling
- Round-robin key rotation
- Usage tracking for observability
"""

from curio_agent_sdk.llm.models import LLMConfig, LLMResponse
from curio_agent_sdk.llm.service import (
    LLMService,
    call_llm,
    call_llm_detailed,
    initialize_llm_service,
    get_llm_service,
)
from curio_agent_sdk.llm.routing import (
    LLMRoutingConfig,
    TierConfig,
    ModelPriority,
    ProviderConfig,
    ModelConfig,
    ProviderKey,
    KeyStatus,
)

__all__ = [
    # Models
    "LLMConfig",
    "LLMResponse",

    # Service
    "LLMService",
    "call_llm",
    "call_llm_detailed",
    "initialize_llm_service",
    "get_llm_service",

    # Routing
    "LLMRoutingConfig",
    "TierConfig",
    "ModelPriority",
    "ProviderConfig",
    "ModelConfig",
    "ProviderKey",
    "KeyStatus",
]
