"""
Unit tests for curio_agent_sdk.core.tools.registry

Covers: ToolRegistry register/get/has/remove, names, tools, llm_schemas
"""

import pytest

from curio_agent_sdk.core.tools.tool import Tool, tool
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.models.exceptions import ToolNotFoundError


def _make_tool(name: str = "test_tool") -> Tool:
    def fn(x: str) -> str:
        """Test."""
        return x

    return Tool(func=fn, name=name)


# ===================================================================
# ToolRegistry
# ===================================================================


class TestToolRegistry:
    def test_empty_registry(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        assert reg.tools == []
        assert reg.names == []

    def test_init_with_tools(self):
        tools = [_make_tool("a"), _make_tool("b")]
        reg = ToolRegistry(tools=tools)
        assert len(reg) == 2
        assert "a" in reg
        assert "b" in reg

    def test_register_tool_instance(self):
        reg = ToolRegistry()
        t = _make_tool("calc")
        result = reg.register(t)
        assert result is t
        assert reg.has("calc")

    def test_register_function(self):
        reg = ToolRegistry()

        def my_func(x: str) -> str:
            """My func."""
            return x

        result = reg.register(my_func)
        assert isinstance(result, Tool)
        assert reg.has("my_func")

    def test_register_function_with_name(self):
        reg = ToolRegistry()

        def my_func(x: str) -> str:
            """My func."""
            return x

        reg.register(my_func, name="custom")
        assert reg.has("custom")

    def test_register_decorated_tool(self):
        @tool
        def search(query: str) -> str:
            """Search."""
            return "results"

        reg = ToolRegistry()
        reg.register(search)
        assert reg.has("search")

    def test_get_existing(self):
        reg = ToolRegistry()
        t = _make_tool("calc")
        reg.register(t)
        result = reg.get("calc")
        assert result is t

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError) as exc_info:
            reg.get("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_nonexistent_shows_available(self):
        reg = ToolRegistry()
        reg.register(_make_tool("calc"))
        reg.register(_make_tool("search"))
        with pytest.raises(ToolNotFoundError) as exc_info:
            reg.get("unknown")
        err = exc_info.value
        assert "calc" in err.available
        assert "search" in err.available

    def test_has_tool_true(self):
        reg = ToolRegistry()
        reg.register(_make_tool("calc"))
        assert reg.has("calc") is True

    def test_has_tool_false(self):
        reg = ToolRegistry()
        assert reg.has("nonexistent") is False

    def test_remove_existing(self):
        reg = ToolRegistry()
        reg.register(_make_tool("calc"))
        result = reg.remove("calc")
        assert result is True
        assert reg.has("calc") is False

    def test_remove_nonexistent(self):
        reg = ToolRegistry()
        result = reg.remove("nonexistent")
        assert result is False

    def test_tools_property(self):
        reg = ToolRegistry()
        t1 = _make_tool("a")
        t2 = _make_tool("b")
        reg.register(t1)
        reg.register(t2)
        tools = reg.tools
        assert len(tools) == 2
        assert t1 in tools
        assert t2 in tools

    def test_names_property(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        names = reg.names
        assert "a" in names
        assert "b" in names

    def test_duplicate_name_overwrites(self):
        reg = ToolRegistry()
        t1 = _make_tool("calc")
        t2 = _make_tool("calc")
        reg.register(t1)
        reg.register(t2)
        assert len(reg) == 1
        assert reg.get("calc") is t2

    def test_get_llm_schemas(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        schemas = reg.get_llm_schemas()
        assert len(schemas) == 2
        schema_names = [s.name for s in schemas]
        assert "a" in schema_names
        assert "b" in schema_names

    def test_get_llm_schemas_empty(self):
        reg = ToolRegistry()
        assert reg.get_llm_schemas() == []

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(_make_tool("a"))
        assert len(reg) == 1
        reg.register(_make_tool("b"))
        assert len(reg) == 2

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        assert "a" in reg
        assert "b" not in reg

    def test_iter(self):
        reg = ToolRegistry()
        t1 = _make_tool("a")
        t2 = _make_tool("b")
        reg.register(t1)
        reg.register(t2)
        tools = list(reg)
        assert len(tools) == 2

    def test_repr(self):
        reg = ToolRegistry()
        reg.register(_make_tool("calc"))
        r = repr(reg)
        assert "ToolRegistry" in r
        assert "calc" in r
