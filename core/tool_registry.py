"""
Tool Registry for Agent Tool Management.

This module provides the ToolRegistry class and the @tool decorator for
managing agent tools. Tools are the actions that an agent can execute,
and this module provides a clean interface for registering, documenting,
and invoking tools.

Example:
    >>> from curio_agent_sdk import ToolRegistry, tool
    >>>
    >>> registry = ToolRegistry()
    >>>
    >>> # Register a tool using decorator
    >>> @registry.tool
    ... def search_web(query: str, max_results: int = 10) -> dict:
    ...     '''Search the web for information.
    ...
    ...     Args:
    ...         query: The search query
    ...         max_results: Maximum number of results to return
    ...
    ...     Returns:
    ...         Search results dictionary
    ...     '''
    ...     # Implementation
    ...     return {"results": [...]}
    >>>
    >>> # Or register programmatically
    >>> registry.register("my_tool", my_function, description="Does something")
    >>>
    >>> # Execute a tool
    >>> result = registry.execute("search_web", {"query": "AI news", "max_results": 5})
"""

from typing import Dict, Any, Callable, Optional, List, TypeVar, Union
from dataclasses import dataclass, field
from functools import wraps
import inspect
import logging
import json

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ToolDefinition:
    """
    Complete definition of a registered tool.

    Attributes:
        name: Unique name/identifier for the tool
        function: The callable that implements the tool
        description: Human-readable description
        parameters: Documentation of parameters
        required_parameters: List of required parameter names
        response_format: Description of the return value
        examples: Usage examples
        enabled: Whether the tool is currently enabled
        metadata: Additional metadata
    """
    name: str
    function: Callable
    description: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    response_format: str = ""
    examples: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_docstring(self) -> str:
        """
        Generate a complete docstring for the tool.

        This is used to provide tool information to the LLM.
        """
        lines = [f"name: {self.name}"]

        if self.description:
            lines.append(f"description: {self.description}")

        if self.parameters:
            lines.append("parameters:")
            for param, desc in self.parameters.items():
                lines.append(f"    {param}: {desc}")

        if self.required_parameters:
            lines.append(f"required_parameters:")
            for param in self.required_parameters:
                lines.append(f"    - {param}")

        if self.response_format:
            lines.append(f"response_format:")
            lines.append(f"    {self.response_format}")

        if self.examples:
            lines.append("examples:")
            for example in self.examples:
                lines.append(f"    >>> {example}")

        return "\n".join(lines)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, str]] = None,
    required_parameters: Optional[List[str]] = None,
    response_format: Optional[str] = None,
    examples: Optional[List[str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark a method as a tool.

    Can be used with or without arguments. If used without arguments,
    the tool name is derived from the function name and other details
    from the docstring.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring first line)
        parameters: Parameter documentation (defaults to parsed docstring)
        required_parameters: Required parameters (defaults to non-default params)
        response_format: Response format description
        examples: Usage examples

    Example:
        >>> @tool
        ... def my_tool(arg1: str) -> dict:
        ...     '''A simple tool.'''
        ...     return {"result": arg1}
        >>>
        >>> @tool(name="search", description="Search for items")
        ... def search_items(query: str, limit: int = 10) -> dict:
        ...     return {"items": [...]}
    """
    def decorator(func: F) -> F:
        # Store tool metadata on the function
        func._tool_metadata = {
            "name": name or func.__name__,
            "description": description,
            "parameters": parameters,
            "required_parameters": required_parameters,
            "response_format": response_format,
            "examples": examples,
        }
        return func
    return decorator


class ToolRegistry:
    """
    Registry for managing agent tools.

    The ToolRegistry provides a centralized place to register, manage,
    and execute tools. It supports:
    - Registration via decorator or programmatic API
    - Automatic docstring parsing for tool documentation
    - Tool enabling/disabling
    - Execution with argument validation
    - Generating tool descriptions for LLM prompts

    Example:
        >>> registry = ToolRegistry()
        >>>
        >>> # Register tools
        >>> registry.register("greet", lambda name: f"Hello, {name}!")
        >>>
        >>> # Get tool descriptions for LLM
        >>> descriptions = registry.get_descriptions()
        >>>
        >>> # Execute tools
        >>> result = registry.execute("greet", {"name": "World"})
        >>> print(result)  # "Hello, World!"
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._execution_history: List[Dict[str, Any]] = []

    def register(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, str]] = None,
        required_parameters: Optional[List[str]] = None,
        response_format: Optional[str] = None,
        examples: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolRegistry":
        """
        Register a tool in the registry.

        If the function has docstring, it will be parsed to extract
        description and parameter documentation if not provided.

        Args:
            name: Unique name for the tool
            function: The callable to register
            description: Tool description (optional, parsed from docstring)
            parameters: Parameter documentation
            required_parameters: Required parameters
            response_format: Response format description
            examples: Usage examples
            metadata: Additional metadata

        Returns:
            self for method chaining

        Raises:
            ValueError: If a tool with this name already exists
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already exists, overwriting")

        # Parse docstring if available
        docstring = function.__doc__ or ""
        parsed = self._parse_docstring(docstring)

        # Build parameter list from signature if not provided
        if parameters is None and not parsed["parameters"]:
            sig = inspect.signature(function)
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'args', 'kwargs'):
                    continue
                param_desc = f"({param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'any'})"
                if param.default != inspect.Parameter.empty:
                    param_desc += f" [default: {param.default}]"
                parameters[param_name] = param_desc

        # Determine required parameters if not provided
        if required_parameters is None:
            sig = inspect.signature(function)
            required_parameters = [
                p.name for p in sig.parameters.values()
                if p.name not in ('self', 'args', 'kwargs')
                and p.default == inspect.Parameter.empty
            ]

        tool_def = ToolDefinition(
            name=name,
            function=function,
            description=description or parsed.get("description", ""),
            parameters=parameters or parsed.get("parameters", {}),
            required_parameters=required_parameters or [],
            response_format=response_format or parsed.get("response_format", ""),
            examples=examples or parsed.get("examples", []),
            metadata=metadata or {},
        )

        self._tools[name] = tool_def
        logger.debug(f"Registered tool: {name}")
        return self

    def register_from_method(self, method: Callable, owner: Any = None) -> str:
        """
        Register a tool from a method that has @tool decorator or docstring.

        Args:
            method: The method to register
            owner: The object that owns the method (for binding)

        Returns:
            The registered tool name
        """
        # Check for @tool decorator metadata
        tool_meta = getattr(method, "_tool_metadata", None)

        if tool_meta:
            name = tool_meta["name"]
            self.register(
                name=name,
                function=method,
                description=tool_meta.get("description"),
                parameters=tool_meta.get("parameters"),
                required_parameters=tool_meta.get("required_parameters"),
                response_format=tool_meta.get("response_format"),
                examples=tool_meta.get("examples"),
            )
        else:
            # Use function name and docstring
            name = method.__name__
            self.register(name=name, function=method)

        return name

    def tool(self, func: F) -> F:
        """
        Decorator to register a function as a tool.

        Example:
            >>> registry = ToolRegistry()
            >>>
            >>> @registry.tool
            ... def my_function(arg: str) -> str:
            ...     '''Description here.'''
            ...     return f"Result: {arg}"
        """
        self.register_from_method(func)
        return func

    def get(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.

        Args:
            name: The tool name

        Returns:
            ToolDefinition or None if not found
        """
        return self._tools.get(name)

    def execute(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with the given arguments.

        Args:
            name: The tool name to execute
            args: Arguments to pass to the tool
            raise_on_error: If True, raise exceptions instead of returning error dict

        Returns:
            Result dictionary with "status" and "result" or "error" keys

        Example:
            >>> result = registry.execute("search", {"query": "AI"})
            >>> if result["status"] == "ok":
            ...     print(result["result"])
        """
        args = args or {}

        # Get tool
        tool_def = self._tools.get(name)
        if tool_def is None:
            error_msg = f"Tool '{name}' not found. Available: {list(self._tools.keys())}"
            logger.error(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return {"status": "error", "result": error_msg}

        if not tool_def.enabled:
            error_msg = f"Tool '{name}' is disabled"
            logger.warning(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return {"status": "error", "result": error_msg}

        # Validate required parameters
        missing = [p for p in tool_def.required_parameters if p not in args]
        if missing:
            error_msg = f"Missing required parameters for '{name}': {missing}"
            logger.error(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return {"status": "error", "result": error_msg}

        # Execute
        try:
            result = tool_def.function(args)

            # Track execution
            self._execution_history.append({
                "tool": name,
                "args": args,
                "status": "success",
            })

            # Normalize result format
            if isinstance(result, dict) and "status" in result:
                return result
            return {"status": "ok", "result": result}

        except Exception as e:
            error_msg = f"Error executing '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)

            self._execution_history.append({
                "tool": name,
                "args": args,
                "status": "error",
                "error": str(e),
            })

            if raise_on_error:
                raise
            return {"status": "error", "result": error_msg}

    def enable(self, name: str) -> bool:
        """Enable a tool."""
        if name in self._tools:
            self._tools[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a tool."""
        if name in self._tools:
            self._tools[name].enabled = False
            return True
        return False

    def remove(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_names(self, enabled_only: bool = True) -> List[str]:
        """
        Get list of registered tool names.

        Args:
            enabled_only: If True, only return enabled tools

        Returns:
            List of tool names
        """
        if enabled_only:
            return [name for name, tool in self._tools.items() if tool.enabled]
        return list(self._tools.keys())

    def get_description(self, name: str) -> Optional[str]:
        """Get the description/docstring for a tool."""
        tool_def = self._tools.get(name)
        if tool_def:
            return tool_def.get_docstring()
        return None

    def get_descriptions(self, enabled_only: bool = True) -> Dict[str, str]:
        """
        Get all tool descriptions as a dictionary.

        Args:
            enabled_only: If True, only include enabled tools

        Returns:
            Dictionary mapping tool names to their docstrings
        """
        return {
            name: tool.get_docstring()
            for name, tool in self._tools.items()
            if not enabled_only or tool.enabled
        }

    def get_descriptions_text(self, enabled_only: bool = True) -> str:
        """
        Get all tool descriptions as formatted text for prompts.

        Args:
            enabled_only: If True, only include enabled tools

        Returns:
            Formatted string with all tool descriptions
        """
        descriptions = self.get_descriptions(enabled_only)
        return "\n\n".join(descriptions.values())

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the tool execution history.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of execution records
        """
        history = list(reversed(self._execution_history))
        if limit:
            return history[:limit]
        return history

    def clear_history(self):
        """Clear the execution history."""
        self._execution_history.clear()

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """
        Parse a docstring to extract tool documentation.

        Supports a custom format with name:, description:, parameters:,
        required_parameters:, response_format:, and examples: sections.
        """
        result = {
            "description": "",
            "parameters": {},
            "required_parameters": [],
            "response_format": "",
            "examples": [],
        }

        if not docstring:
            return result

        lines = docstring.strip().split("\n")
        current_section = None
        current_indent = 0

        for line in lines:
            stripped = line.strip()

            # Check for section headers
            if stripped.startswith("name:"):
                continue  # Skip name, we use the function name
            elif stripped.startswith("description:"):
                result["description"] = stripped[len("description:"):].strip()
                current_section = "description"
            elif stripped.startswith("parameters:"):
                current_section = "parameters"
            elif stripped.startswith("required_parameters:"):
                current_section = "required_parameters"
            elif stripped.startswith("response_format:"):
                result["response_format"] = stripped[len("response_format:"):].strip()
                current_section = "response_format"
            elif stripped.startswith("examples:"):
                current_section = "examples"
            elif stripped.startswith(">>>"):
                if current_section == "examples":
                    result["examples"].append(stripped[3:].strip())
            elif current_section == "parameters" and ":" in stripped:
                # Parameter line: "param_name: description"
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    param_name = parts[0].strip().lstrip("-").strip()
                    param_desc = parts[1].strip()
                    result["parameters"][param_name] = param_desc
            elif current_section == "required_parameters" and stripped.startswith("-"):
                result["required_parameters"].append(stripped.lstrip("-").strip())
            elif current_section == "description" and stripped and not stripped.endswith(":"):
                result["description"] += " " + stripped

        # If no structured format found, use first line as description
        if not result["description"] and lines:
            result["description"] = lines[0].strip()

        return result

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools.keys())
