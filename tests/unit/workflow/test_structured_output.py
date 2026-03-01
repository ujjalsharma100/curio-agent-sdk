"""
Unit tests for curio_agent_sdk.core.workflow.structured_output — response_format_to_schema,
parse_structured_output, Pydantic schema and parsing.
"""

import json
import pytest

from curio_agent_sdk.core.workflow.structured_output import (
    response_format_to_schema,
    parse_structured_output,
)

# Pydantic is in test dependencies
pytest.importorskip("pydantic")
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Pydantic models for tests
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    """Simple flat model."""
    name: str
    count: int = 0


class NestedModel(BaseModel):
    """Model with nested object."""
    title: str
    inner: SimpleModel


# ---------------------------------------------------------------------------
# response_format_to_schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResponseFormatToSchema:
    def test_schema_from_pydantic_model(self):
        """Convert Pydantic model to JSON schema."""
        schema = response_format_to_schema(SimpleModel)
        assert schema["type"] == "json_schema"
        assert "json_schema" in schema
        js = schema["json_schema"]
        assert js["name"] == "SimpleModel"
        assert js["strict"] is True
        assert "schema" in js
        assert "properties" in js["schema"]
        assert "name" in js["schema"]["properties"]
        assert "count" in js["schema"]["properties"]

    def test_schema_from_dict(self):
        """Pass-through dict schema."""
        custom = {"type": "json_object"}
        assert response_format_to_schema(custom) is custom
        custom2 = {"type": "json_schema", "json_schema": {"name": "Custom", "schema": {}}}
        assert response_format_to_schema(custom2) == custom2


# ---------------------------------------------------------------------------
# parse_structured_output
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseStructuredOutput:
    def test_parse_structured_simple(self):
        """Parse JSON into single Pydantic model."""
        text = '{"name": "foo", "count": 42}'
        out = parse_structured_output(text, SimpleModel)
        assert isinstance(out, SimpleModel)
        assert out.name == "foo"
        assert out.count == 42

    def test_parse_structured_nested(self):
        """Nested model parsing."""
        text = '{"title": "T", "inner": {"name": "n", "count": 1}}'
        out = parse_structured_output(text, NestedModel)
        assert isinstance(out, NestedModel)
        assert out.title == "T"
        assert out.inner.name == "n"
        assert out.inner.count == 1

    def test_parse_structured_list(self):
        """Parse list of models (list[SimpleModel])."""
        text = '[{"name": "a", "count": 1}, {"name": "b"}]'
        out = parse_structured_output(text, list[SimpleModel])
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].name == "a" and out[0].count == 1
        assert out[1].name == "b" and out[1].count == 0
        # Also accept object with "items"
        text_items = '{"items": [{"name": "x"}]}'
        out2 = parse_structured_output(text_items, list[SimpleModel])
        assert len(out2) == 1 and out2[0].name == "x"

    def test_parse_structured_invalid(self):
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            parse_structured_output("not json at all", SimpleModel)
        with pytest.raises(json.JSONDecodeError):
            parse_structured_output("{ broken }", SimpleModel)

    def test_parse_structured_missing_fields(self):
        """Missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            parse_structured_output('{"count": 1}', SimpleModel)  # missing "name"

    def test_parse_structured_strips_markdown(self):
        """Markdown code blocks are stripped before parsing."""
        text = '```json\n{"name": "bar", "count": 0}\n```'
        out = parse_structured_output(text, SimpleModel)
        assert out.name == "bar"
        text2 = '```\n{"name": "baz"}\n```'
        out2 = parse_structured_output(text2, SimpleModel)
        assert out2.name == "baz"

    def test_parse_structured_dict_format_returns_raw(self):
        """When response_format is dict, return parsed JSON as-is."""
        text = '{"a": 1, "b": [2, 3]}'
        out = parse_structured_output(text, {"type": "json_object"})
        assert out == {"a": 1, "b": [2, 3]}
