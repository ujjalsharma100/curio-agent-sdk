"""
Tool registry for managing collections of tools.
"""

from __future__ import annotations

from typing import Callable

from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.core.tools.schema import ToolSchema
from curio_agent_sdk.models.llm import ToolSchema as LLMToolSchema
from curio_agent_sdk.exceptions import ToolNotFoundError


class ToolRegistry:
    """
    Registry for managing agent tools.

    Provides lookup, validation, and schema generation for a collection of tools.
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool_or_func: Tool | Callable, name: str | None = None, **kwargs) -> Tool:
        """
        Register a tool. Accepts a Tool instance or a callable.

        Args:
            tool_or_func: A Tool instance or a callable to wrap.
            name: Optional name override (for callables).
            **kwargs: Additional args passed to Tool() if wrapping a callable.

        Returns:
            The registered Tool.
        """
        if isinstance(tool_or_func, Tool):
            t = tool_or_func
        else:
            t = Tool(func=tool_or_func, name=name, **kwargs)

        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool:
        """
        Get a tool by name.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name, list(self._tools.keys()))
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def remove(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get_llm_schemas(self) -> list[LLMToolSchema]:
        """Get LLM-compatible schemas for all registered tools."""
        return [t.to_llm_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.names})"
