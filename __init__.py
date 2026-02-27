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
from curio_agent_sdk.core.state import AgentState, StateExtension
from curio_agent_sdk.core.hooks import (
    HookRegistry,
    HookContext,
    HOOK_EVENTS,
    run_shell_hook,
    load_hooks_from_config,
    load_hooks_from_file,
)
from curio_agent_sdk.core.tools.tool import Tool, tool, ToolConfig
from curio_agent_sdk.core.tools.schema import ToolSchema, ToolParameter
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor, ToolResult

# Loops
from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.loops.tool_calling import ToolCallingLoop

# LLM
from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.llm.router import (
    TieredRouter,
    DegradationStrategy,
    ResetAndRetry,
    FallbackToLowerTier,
    RaiseError,
)
from curio_agent_sdk.llm.providers.base import LLMProvider

# Models
from curio_agent_sdk.models.llm import Message, ToolCall, TokenUsage, LLMRequest, LLMResponse
from curio_agent_sdk.models.agent import AgentRunResult, AgentRun, AgentRunStatus
from curio_agent_sdk.models.events import EventType, StreamEvent, AgentEvent

# Middleware
from curio_agent_sdk.middleware.base import Middleware, MiddlewarePipeline
from curio_agent_sdk.middleware.logging_mw import LoggingMiddleware
from curio_agent_sdk.middleware.cost_tracker import CostTracker
from curio_agent_sdk.middleware.rate_limit import RateLimitMiddleware
from curio_agent_sdk.middleware.tracing import TracingMiddleware
from curio_agent_sdk.middleware.guardrails import GuardrailsMiddleware, GuardrailsError

# Human-in-the-loop
from curio_agent_sdk.core.human_input import HumanInputHandler, CLIHumanInput

# CLI harness
from curio_agent_sdk.core.cli import AgentCLI

# Permissions / sandbox
from curio_agent_sdk.core.permissions import (
    PermissionResult,
    PermissionPolicy,
    AllowAll,
    AskAlways,
    AllowReadsAskWrites,
    CompoundPolicy,
    FileSandboxPolicy,
    NetworkSandboxPolicy,
)

# Memory
from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.vector import VectorMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.composite import CompositeMemory
from curio_agent_sdk.memory.working import WorkingMemory
from curio_agent_sdk.memory.episodic import EpisodicMemory, Episode
from curio_agent_sdk.memory.graph import GraphMemory, Triple
from curio_agent_sdk.memory.self_editing import SelfEditingMemory
from curio_agent_sdk.memory.file_memory import FileMemory
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
from curio_agent_sdk.memory.policies import (
    importance_score,
    decay_score,
    combined_relevance,
    summarize_old_memories,
)

# Structured output
from curio_agent_sdk.core.structured_output import (
    response_format_to_schema,
    parse_structured_output,
)

# State persistence
from curio_agent_sdk.core.checkpoint import Checkpoint
from curio_agent_sdk.core.state_store import (
    StateStore,
    InMemoryStateStore,
    FileStateStore,
)

# Exceptions
from curio_agent_sdk.models.exceptions import (
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

# Context
from curio_agent_sdk.core.context import ContextManager
from curio_agent_sdk.core.instructions import (
    InstructionLoader,
    load_instructions_from_file,
)
from curio_agent_sdk.core.skills import Skill, SkillRegistry, get_active_skill_prompts
from curio_agent_sdk.core.subagent import SubagentConfig, AgentOrchestrator
from curio_agent_sdk.core.plan_mode import (
    Plan,
    PlanStep,
    PlanState,
    Todo,
    TodoState,
    TodoManager,
    PlanMode,
    get_plan_mode_tools,
)
from curio_agent_sdk.core.session import (
    Session,
    SessionManager,
    SessionStore,
    InMemorySessionStore,
)
from curio_agent_sdk.core.task_manager import (
    TaskManager,
    TaskStatus,
    TaskInfo,
    RecoveredRun,
)
from curio_agent_sdk.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)
from curio_agent_sdk.core.plugins import (
    Plugin,
    apply_plugins_to_builder,
    discover_plugins,
)
from curio_agent_sdk.tools.computer_use import ComputerUseToolkit
from curio_agent_sdk.tools.browser import BrowserToolkit
from curio_agent_sdk.mcp import (
    MCPClient,
    MCPToolAdapter,
    MCPBridge,
    MCPServerConfig,
    load_mcp_servers_from_file,
    resolve_env_in_config,
)
from curio_agent_sdk.connectors import (
    Connector,
    ConnectorResource,
    ConnectorBridge,
    resolve_credentials,
)

# Event bus (distributed event streaming)
from curio_agent_sdk.core.event_bus import (
    EventBus,
    InMemoryEventBus,
    EventBusBridge,
    EventFilter,
    DeadLetterEntry,
)

__version__ = "0.6.0"

__all__ = [
    # Core
    "Agent",
    "Runtime",
    "AgentBuilder",
    "AgentState",
    "StateExtension",
    "AgentCLI",
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
    # LLM
    "LLMClient",
    "TieredRouter",
    "DegradationStrategy",
    "ResetAndRetry",
    "FallbackToLowerTier",
    "RaiseError",
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
    # Middleware
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "CostTracker",
    "RateLimitMiddleware",
    "TracingMiddleware",
    "GuardrailsMiddleware",
    "GuardrailsError",
    # Human-in-the-loop
    "HumanInputHandler",
    "CLIHumanInput",
    "PermissionResult",
    "PermissionPolicy",
    "AllowAll",
    "AskAlways",
    "AllowReadsAskWrites",
    "CompoundPolicy",
    "FileSandboxPolicy",
    "NetworkSandboxPolicy",
    # Memory
    "Memory",
    "MemoryEntry",
    "ConversationMemory",
    "VectorMemory",
    "KeyValueMemory",
    "CompositeMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "Episode",
    "GraphMemory",
    "Triple",
    "SelfEditingMemory",
    "FileMemory",
    "importance_score",
    "decay_score",
    "combined_relevance",
    "summarize_old_memories",
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
    # State persistence
    "Checkpoint",
    "StateStore",
    "InMemoryStateStore",
    "FileStateStore",
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
    # Hooks / lifecycle
    "HookRegistry",
    "HookContext",
    "HOOK_EVENTS",
    "run_shell_hook",
    "load_hooks_from_config",
    "load_hooks_from_file",
    # Rules / instructions
    "InstructionLoader",
    "load_instructions_from_file",
    # Skills
    "Skill",
    "SkillRegistry",
    "get_active_skill_prompts",
    # Subagent / multi-agent
    "SubagentConfig",
    "AgentOrchestrator",
    # Plan mode & todos
    "Plan",
    "PlanStep",
    "PlanState",
    "Todo",
    "TodoState",
    "TodoManager",
    "PlanMode",
    "get_plan_mode_tools",
    # Session / conversation management
    "Session",
    "SessionManager",
    "SessionStore",
    "InMemorySessionStore",
    # Long-running task management
    "TaskManager",
    "TaskStatus",
    "TaskInfo",
    "RecoveredRun",
    # Computer use & browser
    "ComputerUseToolkit",
    "BrowserToolkit",
    # MCP (Model Context Protocol)
    "MCPClient",
    "MCPToolAdapter",
    "MCPBridge",
    "MCPServerConfig",
    "load_mcp_servers_from_file",
    "resolve_env_in_config",
    # Connector framework
    "Connector",
    "ConnectorResource",
    "ConnectorBridge",
    "resolve_credentials",
    # Structured output
    "response_format_to_schema",
    "parse_structured_output",
    # Utilities
    "ContextManager",
    # Plugin system
    "Plugin",
    "apply_plugins_to_builder",
    "discover_plugins",
    # Event bus (distributed event streaming)
    "EventBus",
    "InMemoryEventBus",
    "EventBusBridge",
    "EventFilter",
    "DeadLetterEntry",
    # Reliability
    "CircuitBreaker",
    "CircuitState",
]
