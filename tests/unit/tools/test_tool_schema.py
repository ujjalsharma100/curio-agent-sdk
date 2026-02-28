"""
Unit tests for curio_agent_sdk.core.tools.schema

Covers: ToolParameter, ToolSchema, from_function, from_json_schema, validate
"""

import inspect
import pytest

from curio_agent_sdk.core.tools.schema import (
    ToolParameter,
    ToolSchema,
    _parse_docstring,
    _python_type_to_json,
)
from curio_agent_sdk.models.llm import ToolSchema as LLMToolSchema


# ===================================================================
# ToolParameter
# ===================================================================


class TestToolParameter:
    def test_string_param(self):
        p = ToolParameter(name="query", type="string", description="Search query")
        assert p.name == "query"
        assert p.type == "string"
        assert p.description == "Search query"
        assert p.required is True

    def test_integer_param(self):
        p = ToolParameter(name="count", type="integer")
        assert p.type == "integer"

    def test_boolean_param(self):
        p = ToolParameter(name="active", type="boolean")
        assert p.type == "boolean"

    def test_number_param(self):
        p = ToolParameter(name="price", type="number")
        assert p.type == "number"

    def test_array_param(self):
        p = ToolParameter(name="tags", type="array", items={"type": "string"})
        assert p.type == "array"
        assert p.items == {"type": "string"}

    def test_enum_param(self):
        p = ToolParameter(name="color", type="string", enum=["red", "green", "blue"])
        assert p.enum == ["red", "green", "blue"]

    def test_optional_param(self):
        p = ToolParameter(name="limit", type="integer", required=False, default=10)
        assert p.required is False
        assert p.default == 10

    def test_to_json_schema_basic(self):
        p = ToolParameter(name="q", type="string", description="Query")
        schema = p.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "Query"
        assert "enum" not in schema
        assert "items" not in schema

    def test_to_json_schema_with_enum(self):
        p = ToolParameter(name="sort", type="string", enum=["asc", "desc"])
        schema = p.to_json_schema()
        assert schema["enum"] == ["asc", "desc"]

    def test_to_json_schema_with_items(self):
        p = ToolParameter(name="ids", type="array", items={"type": "integer"})
        schema = p.to_json_schema()
        assert schema["items"] == {"type": "integer"}

    def test_to_json_schema_with_default(self):
        p = ToolParameter(name="limit", type="integer", required=False, default=10)
        schema = p.to_json_schema()
        assert schema["default"] == 10

    def test_to_json_schema_no_description(self):
        p = ToolParameter(name="x", type="string")
        schema = p.to_json_schema()
        assert "description" not in schema

    def test_default_sentinel(self):
        p = ToolParameter(name="x", type="string")
        assert p.default is inspect.Parameter.empty


# ===================================================================
# ToolSchema
# ===================================================================


class TestToolSchema:
    def test_creation(self):
        schema = ToolSchema(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )
        assert schema.name == "search"
        assert schema.description == "Search the web"
        assert len(schema.parameters) == 1

    def test_to_json_schema(self):
        schema = ToolSchema(
            name="calc",
            description="Calculate",
            parameters=[
                ToolParameter(name="expression", type="string", description="Math expr", required=True),
                ToolParameter(name="precision", type="integer", required=False, default=2),
            ],
        )
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert "expression" in js["properties"]
        assert "precision" in js["properties"]
        assert "expression" in js["required"]
        assert "precision" not in js["required"]
        assert js["additionalProperties"] is False

    def test_to_json_schema_no_required(self):
        schema = ToolSchema(
            name="fn",
            description="Fn",
            parameters=[
                ToolParameter(name="x", type="string", required=False, default="a"),
            ],
        )
        js = schema.to_json_schema()
        assert "required" not in js

    def test_to_json_schema_empty_params(self):
        schema = ToolSchema(name="noop", description="No-op", parameters=[])
        js = schema.to_json_schema()
        assert js["properties"] == {}
        assert js["additionalProperties"] is False

    def test_to_llm_schema(self):
        schema = ToolSchema(
            name="search",
            description="Search",
            parameters=[
                ToolParameter(name="q", type="string", description="Query"),
            ],
        )
        llm = schema.to_llm_schema()
        assert isinstance(llm, LLMToolSchema)
        assert llm.name == "search"
        assert llm.description == "Search"
        assert llm.parameters["type"] == "object"
        assert "q" in llm.parameters["properties"]

    def test_validate_valid_args(self):
        schema = ToolSchema(
            name="fn",
            description="Fn",
            parameters=[
                ToolParameter(name="a", type="string"),
                ToolParameter(name="b", type="integer"),
            ],
        )
        result = schema.validate({"a": "hello", "b": 42})
        assert result == {"a": "hello", "b": 42}

    def test_validate_with_default(self):
        schema = ToolSchema(
            name="fn",
            description="Fn",
            parameters=[
                ToolParameter(name="a", type="string"),
                ToolParameter(name="b", type="integer", required=False, default=10),
            ],
        )
        result = schema.validate({"a": "hello"})
        assert result == {"a": "hello", "b": 10}

    def test_validate_missing_required(self):
        schema = ToolSchema(
            name="fn",
            description="Fn",
            parameters=[
                ToolParameter(name="a", type="string"),
                ToolParameter(name="b", type="string"),
            ],
        )
        from curio_agent_sdk.models.exceptions import ToolValidationError

        with pytest.raises(ToolValidationError) as exc_info:
            schema.validate({"a": "hello"})
        assert "b" in str(exc_info.value)

    def test_validate_extra_args_ignored(self):
        schema = ToolSchema(
            name="fn",
            description="Fn",
            parameters=[
                ToolParameter(name="a", type="string"),
            ],
        )
        result = schema.validate({"a": "hello", "extra": "ignored"})
        assert result == {"a": "hello"}

    # ---- from_function ----

    def test_from_function_simple(self):
        def greet(name: str) -> str:
            """Greet someone.

            Args:
                name: The person's name
            """
            return f"Hello, {name}!"

        schema = ToolSchema.from_function(greet)
        assert schema.name == "greet"
        assert "Greet someone." in schema.description
        assert len(schema.parameters) == 1
        assert schema.parameters[0].name == "name"
        assert schema.parameters[0].type == "string"
        assert schema.parameters[0].required is True

    def test_from_function_complex(self):
        def search(query: str, limit: int = 10, tags: list[str] | None = None) -> str:
            """Search for items.

            Args:
                query: Search query
                limit: Max results
                tags: Filter tags
            """
            return ""

        schema = ToolSchema.from_function(search)
        assert schema.name == "search"
        param_map = {p.name: p for p in schema.parameters}
        assert param_map["query"].required is True
        assert param_map["query"].type == "string"
        assert param_map["limit"].required is False
        assert param_map["limit"].default == 10
        assert param_map["limit"].type == "integer"
        assert "tags" in param_map

    def test_from_function_with_name_override(self):
        def fn(x: int) -> int:
            """Do stuff."""
            return x

        schema = ToolSchema.from_function(fn, name="custom_name")
        assert schema.name == "custom_name"

    def test_from_function_with_description_override(self):
        def fn(x: int) -> int:
            """Original."""
            return x

        schema = ToolSchema.from_function(fn, description="Custom desc")
        assert schema.description == "Custom desc"

    def test_from_function_no_hints(self):
        def fn(x, y):
            """No hints."""
            return x + y

        schema = ToolSchema.from_function(fn)
        assert len(schema.parameters) == 2
        for p in schema.parameters:
            assert p.type == "string"  # default when no hints

    def test_from_function_no_docstring(self):
        def fn(x: int) -> int:
            return x

        schema = ToolSchema.from_function(fn)
        assert schema.description  # should have some description
        assert len(schema.parameters) == 1

    def test_from_function_skips_self_cls(self):
        def method(self, x: int) -> int:
            """Method."""
            return x

        schema = ToolSchema.from_function(method)
        param_names = [p.name for p in schema.parameters]
        assert "self" not in param_names
        assert "x" in param_names

    # ---- from_json_schema ----

    def test_from_json_schema(self):
        json_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        }
        schema = ToolSchema.from_json_schema("search", "Search", json_schema)
        assert schema.name == "search"
        assert schema.description == "Search"
        param_map = {p.name: p for p in schema.parameters}
        assert param_map["query"].required is True
        assert param_map["limit"].required is False

    def test_from_json_schema_with_enum(self):
        json_schema = {
            "type": "object",
            "properties": {
                "sort": {"type": "string", "enum": ["asc", "desc"]},
            },
            "required": ["sort"],
        }
        schema = ToolSchema.from_json_schema("fn", "Fn", json_schema)
        assert schema.parameters[0].enum == ["asc", "desc"]

    def test_from_json_schema_with_array(self):
        json_schema = {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
            },
        }
        schema = ToolSchema.from_json_schema("fn", "Fn", json_schema)
        assert schema.parameters[0].type == "array"
        assert schema.parameters[0].items == {"type": "integer"}

    def test_from_json_schema_empty(self):
        schema = ToolSchema.from_json_schema("fn", "Fn", {"type": "object"})
        assert schema.parameters == []


# ===================================================================
# Helper functions
# ===================================================================


class TestPythonTypeToJson:
    def test_str(self):
        assert _python_type_to_json(str) == "string"

    def test_int(self):
        assert _python_type_to_json(int) == "integer"

    def test_float(self):
        assert _python_type_to_json(float) == "number"

    def test_bool(self):
        assert _python_type_to_json(bool) == "boolean"

    def test_list(self):
        assert _python_type_to_json(list) == "array"

    def test_dict(self):
        assert _python_type_to_json(dict) == "object"

    def test_none(self):
        assert _python_type_to_json(None) == "string"

    def test_unknown(self):
        class Custom:
            pass

        assert _python_type_to_json(Custom) == "string"

    def test_list_generic(self):
        assert _python_type_to_json(list[str]) == "array"

    def test_dict_generic(self):
        assert _python_type_to_json(dict[str, int]) == "object"

    def test_optional(self):
        assert _python_type_to_json(str | None) == "string"


class TestParseDocstring:
    def test_simple(self):
        desc, params = _parse_docstring("Do something.")
        assert desc == "Do something."
        assert params == {}

    def test_with_args(self):
        doc = """Do something.

        Args:
            x: The x value
            y: The y value
        """
        desc, params = _parse_docstring(doc)
        assert "Do something." in desc
        assert params["x"] == "The x value"
        assert params["y"] == "The y value"

    def test_with_returns(self):
        doc = """Do something.

        Args:
            x: The x value

        Returns:
            The result
        """
        desc, params = _parse_docstring(doc)
        assert params["x"] == "The x value"

    def test_empty(self):
        desc, params = _parse_docstring("")
        assert desc == ""
        assert params == {}

    def test_multiline_desc(self):
        doc = """First line.
        Second line.

        Args:
            x: Value
        """
        desc, params = _parse_docstring(doc)
        assert "First line." in desc
        assert "Second line." in desc
