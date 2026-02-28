"""
Core module for Curio Agent SDK.

Contains the agent, loops, tools, and state management.
"""

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.component import Component
from curio_agent_sdk.core.state import AgentState, StateExtension
from curio_agent_sdk.core.tools import Tool, tool, ToolSchema, ToolRegistry, ToolExecutor
from curio_agent_sdk.core.loops import AgentLoop, ToolCallingLoop
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
from curio_agent_sdk.core.structured_output import (
    response_format_to_schema,
    parse_structured_output,
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
)
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
from curio_agent_sdk.core.plugins import (
    Plugin,
    apply_plugins_to_builder,
    discover_plugins,
)
from curio_agent_sdk.core.event_bus import (
    EventBus,
    InMemoryEventBus,
    EventBusBridge,
    EventFilter,
    DeadLetterEntry,
)
from curio_agent_sdk.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)

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
