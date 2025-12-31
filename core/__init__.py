"""
Core module for Curio Agent SDK.

This module contains the fundamental building blocks:
- BaseAgent: Abstract base class for all agents
- ObjectIdentifierMap: Context window optimization through object abstraction
- ToolRegistry: Flexible tool registration and management
- Models: Core data models for agent runs and events
"""

from curio_agent_sdk.core.base_agent import BaseAgent
from curio_agent_sdk.core.object_identifier_map import ObjectIdentifierMap
from curio_agent_sdk.core.tool_registry import ToolRegistry, tool
from curio_agent_sdk.core.models import AgentRun, AgentRunEvent, AgentLLMUsage

__all__ = [
    "BaseAgent",
    "ObjectIdentifierMap",
    "ToolRegistry",
    "tool",
    "AgentRun",
    "AgentRunEvent",
    "AgentLLMUsage",
]
