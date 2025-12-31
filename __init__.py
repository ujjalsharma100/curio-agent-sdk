"""
Curio Agent SDK - A flexible, model-agnostic agentic framework.

This SDK provides a comprehensive toolkit for building autonomous agents with:
- Model-agnostic LLM calling with provider abstraction
- Tiered model routing with automatic failover and health tracking
- Object identifier maps for context window optimization
- Flexible tool registry with decorator and docstring support
- Plan-critique-synthesize agentic loop
- Database persistence for observability
- Highly configurable and extensible architecture

Example:
    from curio_agent_sdk import BaseAgent, AgentConfig, tool

    # Configure the SDK
    config = AgentConfig.from_env()

    # Create your custom agent
    class MyAgent(BaseAgent):
        def get_agent_instructions(self):
            return '''
            You are a helpful assistant.

            ## GUIDELINES
            - Be concise and accurate
            '''

        def initialize_tools(self):
            self.tool_registry.register_from_method(self.greet)

        @tool(name="greet", description="Greet a user by name")
        def greet(self, args):
            return {"status": "ok", "result": f"Hello, {args.get('name', 'World')}!"}

    # Run your agent
    agent = MyAgent(agent_id="my-agent", config=config)
    result = agent.run(objective="Greet the user Alice")
"""

__version__ = "0.1.0"
__author__ = "Curio Team"

# Core exports
from curio_agent_sdk.core.base_agent import BaseAgent
from curio_agent_sdk.core.object_identifier_map import ObjectIdentifierMap
from curio_agent_sdk.core.tool_registry import ToolRegistry, tool
from curio_agent_sdk.core.models import (
    AgentRun,
    AgentRunEvent,
    AgentLLMUsage,
)

# LLM exports
from curio_agent_sdk.llm.service import LLMService, call_llm, get_llm_service, initialize_llm_service
from curio_agent_sdk.llm.models import LLMConfig, LLMResponse
from curio_agent_sdk.llm.routing import LLMRoutingConfig, TierConfig, ModelPriority, ProviderConfig, ModelConfig

# Provider exports
from curio_agent_sdk.llm.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GroqProvider,
    OllamaProvider,
)

# Persistence exports
from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.persistence.postgres import PostgresPersistence
from curio_agent_sdk.persistence.sqlite import SQLitePersistence
from curio_agent_sdk.persistence.memory import InMemoryPersistence

# Config exports
from curio_agent_sdk.config.settings import AgentConfig, DatabaseConfig

__all__ = [
    # Version
    "__version__",

    # Core
    "BaseAgent",
    "ObjectIdentifierMap",
    "ToolRegistry",
    "tool",
    "AgentRun",
    "AgentRunEvent",
    "AgentLLMUsage",

    # LLM
    "LLMService",
    "call_llm",
    "get_llm_service",
    "initialize_llm_service",
    "LLMConfig",
    "LLMResponse",
    "LLMRoutingConfig",
    "TierConfig",
    "ModelPriority",
    "ProviderConfig",
    "ModelConfig",

    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GroqProvider",
    "OllamaProvider",

    # Persistence
    "BasePersistence",
    "PostgresPersistence",
    "SQLitePersistence",
    "InMemoryPersistence",

    # Config
    "AgentConfig",
    "DatabaseConfig",
]
