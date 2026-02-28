"""
Structured output support: Pydantic model → JSON schema and response parsing.

Enables agent.arun(..., response_format=list[SearchResult]) so that
result.output is raw text and result.parsed_output is list[SearchResult].
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar, get_origin, get_args

T = TypeVar("T")


def _get_pydantic_model_json_schema(model: type) -> dict[str, Any]:
    """Get JSON schema from a Pydantic model. Requires pydantic >= 2."""
    try:
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "Structured output with Pydantic models requires pydantic. "
            "Install with: pip install pydantic"
        ) from None
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError(f"Expected a Pydantic BaseModel subclass, got {type(model)}")
    return model.model_json_schema()


def response_format_to_schema(response_format: type[T] | dict[str, Any]) -> dict[str, Any]:
    """
    Convert a response_format (Pydantic model type, list[Model], or raw dict) to
    a provider-compatible response_format dict for LLMRequest.

    - dict: returned as-is (assumed already in provider format).
    - Pydantic BaseModel: returns OpenAI-style json_schema for that model.
    - list[SomeModel]: returns schema for array of items matching SomeModel.

    Returns a dict suitable for LLMRequest.response_format, e.g.:
    - {"type": "json_object"} for generic JSON
    - {"type": "json_schema", "json_schema": {"name": "...", "strict": True, "schema": {...}}}
    """
    if isinstance(response_format, dict):
        return response_format

    try:
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "Structured output with Pydantic models requires pydantic. "
            "Install with: pip install pydantic"
        ) from None

    origin = get_origin(response_format)
    args = get_args(response_format)

    if origin is list and args:
        # list[SomeModel] -> object with "items" array (OpenAI requires root type "object")
        item_type = args[0]
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            item_schema = _get_pydantic_model_json_schema(item_type)
            schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": item_schema,
                        "description": "List of results",
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            }
            name = getattr(item_type, "__name__", "Item")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{name}List",
                    "strict": True,
                    "schema": schema,
                },
            }
        # Fallback: generic json_object
        return {"type": "json_object"}

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema = _get_pydantic_model_json_schema(response_format)
        name = getattr(response_format, "__name__", "Response")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }

    return {"type": "json_object"}


def parse_structured_output(text: str, response_format: type[T] | dict[str, Any]) -> T | list[Any]:
    """
    Parse LLM text output as JSON and validate/coerce into the given type.

    - If response_format is a Pydantic model, returns a single instance.
    - If response_format is list[Model], expects JSON array or object with "items"
      and returns list of model instances.
    - If response_format is a dict (raw schema), returns parsed JSON as dict/list.

    Strips markdown code blocks (```json ... ```) if present.
    """
    raw = _extract_json(text)
    data = json.loads(raw)

    if isinstance(response_format, dict):
        return data

    try:
        from pydantic import BaseModel
    except ImportError:
        return data

    origin = get_origin(response_format)
    args = get_args(response_format)

    if origin is list and args:
        item_type = args[0]
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, list):
                items = data
            else:
                items = [data] if data is not None else []
            return [item_type.model_validate(x) for x in items]
        return data if isinstance(data, list) else [data]

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return response_format.model_validate(data)

    return data


def _extract_json(text: str) -> str:
    """Extract JSON string from text, optionally inside markdown code blocks."""
    text = (text or "").strip()
    # Strip ```json ... ``` or ``` ... ```
    match = re.search(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text
