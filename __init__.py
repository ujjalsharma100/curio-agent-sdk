"""
Curio Agent SDK - Production-grade agent harness.

Build any kind of agent: coding assistants, research agents,
data intelligence tools, personal assistants, and more.

Quick start:
    from curio_agent_sdk import Agent, tool

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return "search results..."

    agent = Agent(
        model="openai:gpt-4o",
        tools=[search],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.run("What are the latest AI developments?")
    print(result.output)
"""

# Core
from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.runtime import Runtime
from curio_agent_sdk.core.builder import AgentBuilder
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool, tool, ToolConfig
from curio_agent_sdk.core.tools.schema import ToolSchema, ToolParameter
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor, ToolResult

# Loops
from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.loops.tool_calling import ToolCallingLoop
from curio_agent_sdk.core.loops.plan_critique import PlanCritiqueSynthesizeLoop

# LLM
from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.llm.router import TieredRouter
from curio_agent_sdk.llm.providers.base import LLMProvider

# Models
from curio_agent_sdk.models.llm import Message, ToolCall, TokenUsage, LLMRequest, LLMResponse
from curio_agent_sdk.models.agent import AgentRunResult, AgentRun, AgentRunStatus
from curio_agent_sdk.models.events import EventType, StreamEvent, AgentEvent

# Config
from curio_agent_sdk.config.settings import AgentConfig, DatabaseConfig

# Persistence
from curio_agent_sdk.persistence.base import BasePersistence

# Middleware
from curio_agent_sdk.middleware.base import Middleware, MiddlewarePipeline
from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.middleware.cost_tracker import CostTracker
from curio_agent_sdk.middleware.retry import RetryMiddleware
from curio_agent_sdk.middleware.rate_limit import RateLimitMiddleware
from curio_agent_sdk.middleware.tracing import TracingMiddleware
from curio_agent_sdk.middleware.guardrails import GuardrailsMiddleware, GuardrailsError

# Human-in-the-loop
from curio_agent_sdk.core.human_input import HumanInputHandler, CLIHumanInput

# Memory
from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.vector import VectorMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.composite import CompositeMemory
from curio_agent_sdk.memory.manager import (
    MemoryManager,
    MemoryInjectionStrategy,
    MemorySaveStrategy,
    MemoryQueryStrategy,
    DefaultInjection,
    UserMessageInjection,
    NoInjection,
    DefaultSave,
    SaveEverythingStrategy,
    NoSave,
    PerIterationSave,
    DefaultQuery,
    KeywordQuery,
    AdaptiveTokenQuery,
)

# Checkpointing
from curio_agent_sdk.core.checkpoint import (
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
    FileCheckpointStore,
)

# Exceptions
from curio_agent_sdk.exceptions import (
    CurioError,
    LLMError,
    LLMRateLimitError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    AgentError,
    AgentTimeoutError,
    NoAvailableModelError,
    CostBudgetExceeded,
)

# Context optimization
from curio_agent_sdk.core.object_identifier_map import ObjectIdentifierMap
from curio_agent_sdk.core.context import ContextManager

__version__ = "0.4.0"

__all__ = [
    # Core
    "Agent",
    "Runtime",
    "AgentBuilder",
    "AgentState",
    "Tool",
    "tool",
    "ToolConfig",
    "ToolSchema",
    "ToolParameter",
    "ToolRegistry",
    "ToolExecutor",
    "ToolResult",
    # Loops
    "AgentLoop",
    "ToolCallingLoop",
    "PlanCritiqueSynthesizeLoop",
    # LLM
    "LLMClient",
    "TieredRouter",
    "LLMProvider",
    "Message",
    "ToolCall",
    "TokenUsage",
    "LLMRequest",
    "LLMResponse",
    # Models
    "AgentRunResult",
    "AgentRun",
    "AgentRunStatus",
    "EventType",
    "StreamEvent",
    "AgentEvent",
    # Config
    "AgentConfig",
    "DatabaseConfig",
    # Persistence
    "BasePersistence",
    # Middleware
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "CostTracker",
    "RetryMiddleware",
    "RateLimitMiddleware",
    "TracingMiddleware",
    "GuardrailsMiddleware",
    "GuardrailsError",
    # Human-in-the-loop
    "HumanInputHandler",
    "CLIHumanInput",
    # Memory
    "Memory",
    "MemoryEntry",
    "ConversationMemory",
    "VectorMemory",
    "KeyValueMemory",
    "CompositeMemory",
    # Memory Manager and Strategies
    "MemoryManager",
    "MemoryInjectionStrategy",
    "MemorySaveStrategy",
    "MemoryQueryStrategy",
    "DefaultInjection",
    "UserMessageInjection",
    "NoInjection",
    "DefaultSave",
    "SaveEverythingStrategy",
    "NoSave",
    "PerIterationSave",
    "DefaultQuery",
    "KeywordQuery",
    "AdaptiveTokenQuery",
    # Checkpointing
    "Checkpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    # Exceptions
    "CurioError",
    "LLMError",
    "LLMRateLimitError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "AgentError",
    "AgentTimeoutError",
    "NoAvailableModelError",
    "CostBudgetExceeded",
    # Utilities
    "ObjectIdentifierMap",
    "ContextManager",
]
