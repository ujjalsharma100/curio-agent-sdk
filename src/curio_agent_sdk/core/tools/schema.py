"""
Tool schema generation for native LLM tool calling.

Generates JSON Schema from Python function signatures, type hints,
and docstrings. Supports Pydantic models for complex parameter types.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from curio_agent_sdk.models.llm import ToolSchema as LLMToolSchema


# Python type -> JSON Schema type mapping
TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolParameter:
    """Schema for a single tool parameter."""
    name: str
    type: str  # JSON Schema type
    description: str = ""
    required: bool = True
    enum: list[str] | None = None
    default: Any = inspect.Parameter.empty
    items: dict | None = None  # For array types

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.items:
            schema["items"] = self.items
        if self.default is not inspect.Parameter.empty:
            schema["default"] = self.default
        return schema


@dataclass
class ToolSchema:
    """
    Complete schema for a tool, used for JSON Schema generation
    and LLM native tool calling.
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Generate JSON Schema for tool parameters."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        # Allow additional properties to be false for strict schemas
        schema["additionalProperties"] = False

        return schema

    def to_llm_schema(self) -> LLMToolSchema:
        """Convert to the LLM-layer ToolSchema for provider APIs."""
        return LLMToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.to_json_schema(),
        )

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """
        Validate arguments against schema. Returns validated args.

        Raises:
            ValueError: If required parameters are missing or types don't match.
        """
        errors = []
        validated = {}

        param_map = {p.name: p for p in self.parameters}

        # Check required params
        for param in self.parameters:
            if param.required and param.name not in args:
                errors.append(f"Missing required parameter: {param.name}")

        if errors:
            from curio_agent_sdk.models.exceptions import ToolValidationError
            raise ToolValidationError(self.name, errors)

        # Apply defaults and pass through
        for param in self.parameters:
            if param.name in args:
                validated[param.name] = args[param.name]
            elif param.default is not inspect.Parameter.empty:
                validated[param.name] = param.default

        return validated

    @classmethod
    def from_json_schema(
        cls,
        name: str,
        description: str,
        json_schema: dict[str, Any],
    ) -> ToolSchema:
        """
        Build a ToolSchema from a JSON Schema object (e.g. MCP inputSchema).

        Uses "properties" and "required" to build ToolParameter list.
        """
        properties = json_schema.get("properties", {})
        required_names = set(json_schema.get("required", []))
        parameters = []
        for param_name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            param_type = prop.get("type", "string")
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=prop.get("description", ""),
                    required=param_name in required_names,
                    enum=prop.get("enum"),
                    items=prop.get("items") if param_type == "array" else None,
                )
            )
        return cls(name=name, description=description, parameters=parameters)

    @classmethod
    def from_function(cls, func: Callable, name: str | None = None, description: str | None = None) -> ToolSchema:
        """
        Auto-generate a ToolSchema from a function's signature, type hints, and docstring.

        Args:
            func: The function to generate schema from.
            name: Override name (defaults to function name).
            description: Override description (defaults to docstring first line).
        """
        func_name = name or func.__name__
        sig = inspect.signature(func)

        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Parse docstring for descriptions
        doc = func.__doc__ or ""
        func_desc, param_docs = _parse_docstring(doc)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine JSON type
            hint = hints.get(param_name)
            json_type = _python_type_to_json(hint)

            # Determine if required
            has_default = param.default is not inspect.Parameter.empty
            is_required = not has_default

            # Get description from docstring
            param_desc = param_docs.get(param_name, "")

            parameters.append(ToolParameter(
                name=param_name,
                type=json_type,
                description=param_desc,
                required=is_required,
                default=param.default if has_default else inspect.Parameter.empty,
            ))

        return cls(
            name=func_name,
            description=description or func_desc or f"Execute {func_name}",
            parameters=parameters,
        )


def _python_type_to_json(hint: Any) -> str:
    """Convert a Python type hint to JSON Schema type string."""
    if hint is None:
        return "string"

    # Handle Optional, Union, etc.
    origin = getattr(hint, "__origin__", None)
    if origin is not None:
        # Handle list[X], List[X]
        if origin is list:
            return "array"
        # Handle dict[X, Y], Dict[X, Y]
        if origin is dict:
            return "object"
        # Handle Optional[X] (Union[X, None])
        args = getattr(hint, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_json(non_none[0])

    return TYPE_MAP.get(hint, "string")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """
    Parse a Google/Numpy-style docstring into description and parameter docs.

    Returns:
        (function_description, {param_name: param_description})
    """
    if not doc:
        return "", {}

    lines = doc.strip().split("\n")
    description_lines = []
    param_docs: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args = True
            continue
        elif stripped.lower() in ("returns:", "return:", "raises:", "yields:", "examples:", "example:", "note:", "notes:"):
            in_args = False
            continue

        if in_args:
            # Try to match "param_name: description" or "param_name (type): description"
            match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if match:
                param_docs[match.group(1)] = match.group(2).strip()
        elif not in_args and not param_docs:
            if stripped:
                description_lines.append(stripped)

    return " ".join(description_lines), param_docs
