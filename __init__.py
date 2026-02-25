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
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool, tool
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

__version__ = "0.2.0"

__all__ = [
    # Core
    "Agent",
    "AgentState",
    "Tool",
    "tool",
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
