"""
Core module for Curio Agent SDK.

Contains the agent, loops, tools, and state management.
"""

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.component import Component
from curio_agent_sdk.core.state import AgentState, StateExtension
from curio_agent_sdk.core.tools import Tool, tool, ToolSchema, ToolRegistry, ToolExecutor
from curio_agent_sdk.core.loops import AgentLoop, ToolCallingLoop, PlanCritiqueSynthesizeLoop

# Preserve ObjectIdentifierMap (still useful for context optimization)
from curio_agent_sdk.core.object_identifier_map import ObjectIdentifierMap
from curio_agent_sdk.core.context import ContextManager
from curio_agent_sdk.core.instructions import (
    InstructionLoader,
    load_instructions_from_file,
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
    "PlanCritiqueSynthesizeLoop",
    "ObjectIdentifierMap",
    "ContextManager",
    "InstructionLoader",
    "load_instructions_from_file",
]
