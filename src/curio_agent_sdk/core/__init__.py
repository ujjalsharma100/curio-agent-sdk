"""
Core module for Curio Agent SDK.

Contains the agent, loops, tools, and state management.
"""

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.base import Component
from curio_agent_sdk.core.tools import Tool, tool, ToolSchema, ToolRegistry, ToolExecutor
from curio_agent_sdk.core.loops import AgentLoop, ToolCallingLoop
from curio_agent_sdk.core.context import (
    ContextManager,
    InstructionLoader,
    load_instructions_from_file,
)
from curio_agent_sdk.core.extensions import (
    Skill,
    SkillRegistry,
    get_active_skill_prompts,
    SubagentConfig,
    AgentOrchestrator,
    Plugin,
    apply_plugins_to_builder,
    discover_plugins,
)
from curio_agent_sdk.core.workflow import (
    Plan,
    PlanStep,
    PlanState,
    Todo,
    TodoState,
    TodoManager,
    PlanMode,
    get_plan_mode_tools,
    response_format_to_schema,
    parse_structured_output,
    TaskManager,
    TaskStatus,
    TaskInfo,
)
from curio_agent_sdk.core.state import (
    AgentState,
    StateExtension,
    StateStore,
    InMemoryStateStore,
    Checkpoint,
    Session,
    SessionManager,
    SessionStore,
    InMemorySessionStore,
)
from curio_agent_sdk.core.security import (
    PermissionResult,
    PermissionPolicy,
    AllowAll,
    AskAlways,
    AllowReadsAskWrites,
    CompoundPolicy,
    FileSandboxPolicy,
    NetworkSandboxPolicy,
    HumanInputHandler,
    CLIHumanInput,
)
from curio_agent_sdk.core.events import (
    EventBus,
    InMemoryEventBus,
    EventBusBridge,
    EventFilter,
    DeadLetterEntry,
)
from curio_agent_sdk.resilience import CircuitBreaker, CircuitState

__all__ = [
    "Agent",
    "Component",
    "AgentState",
    "StateExtension",
    "Tool",
    "tool",
    "ToolSchema",
    "ToolRegistry",
    "ToolExecutor",
    "AgentLoop",
    "ToolCallingLoop",
    "ContextManager",
    "InstructionLoader",
    "load_instructions_from_file",
    "Skill",
    "SkillRegistry",
    "get_active_skill_prompts",
    "SubagentConfig",
    "AgentOrchestrator",
    "Plan",
    "PlanStep",
    "PlanState",
    "Todo",
    "TodoState",
    "TodoManager",
    "PlanMode",
    "get_plan_mode_tools",
    "response_format_to_schema",
    "parse_structured_output",
    "Session",
    "SessionManager",
    "SessionStore",
    "InMemorySessionStore",
    "TaskManager",
    "TaskStatus",
    "TaskInfo",
    "PermissionResult",
    "PermissionPolicy",
    "AllowAll",
    "AskAlways",
    "AllowReadsAskWrites",
    "CompoundPolicy",
    "FileSandboxPolicy",
    "NetworkSandboxPolicy",
    "HumanInputHandler",
    "CLIHumanInput",
    "Plugin",
    "apply_plugins_to_builder",
    "discover_plugins",
    "EventBus",
    "InMemoryEventBus",
    "EventBusBridge",
    "EventFilter",
    "DeadLetterEntry",
    "CircuitBreaker",
    "CircuitState",
]
