"""
Unit tests for curio_agent_sdk.core.tools.tool

Covers: Tool class, @tool decorator, ToolConfig, execution, retry, timeout
"""

import asyncio
import pytest

from curio_agent_sdk.core.tools.tool import Tool, ToolConfig, tool
from curio_agent_sdk.core.tools.schema import ToolSchema
from curio_agent_sdk.models.exceptions import ToolExecutionError, ToolTimeoutError


# ===================================================================
# ToolConfig
# ===================================================================


class TestToolConfig:
    def test_defaults(self):
        cfg = ToolConfig()
        assert cfg.timeout == 60.0
        assert cfg.max_retries == 0
        assert cfg.retry_backoff == 1.0
        assert cfg.require_confirmation is False
        assert cfg.cache_ttl is None
        assert cfg.sandboxed is False
        assert cfg.idempotent is False

    def test_custom(self):
        cfg = ToolConfig(
            timeout=30.0,
            max_retries=3,
            retry_backoff=2.0,
            require_confirmation=True,
            cache_ttl=60.0,
            sandboxed=True,
            idempotent=True,
        )
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3
        assert cfg.retry_backoff == 2.0
        assert cfg.require_confirmation is True
        assert cfg.cache_ttl == 60.0
        assert cfg.sandboxed is True
        assert cfg.idempotent is True


# ===================================================================
# Tool class
# ===================================================================


class TestTool:
    def test_from_sync_function(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = Tool(func=add)
        assert t.name == "add"
        assert t.description == "Add two numbers."
        assert t._is_async is False

    def test_from_async_function(self):
        async def fetch(url: str) -> str:
            """Fetch a URL."""
            return "content"

        t = Tool(func=fetch)
        assert t.name == "fetch"
        assert t._is_async is True

    def test_name_from_function(self):
        def my_tool():
            pass

        t = Tool(func=my_tool)
        assert t.name == "my_tool"

    def test_name_override(self):
        def my_tool():
            pass

        t = Tool(func=my_tool, name="custom_name")
        assert t.name == "custom_name"

    def test_description_from_docstring(self):
        def my_tool():
            """This is the description."""
            pass

        t = Tool(func=my_tool)
        assert t.description == "This is the description."

    def test_description_override(self):
        def my_tool():
            """Original description."""
            pass

        t = Tool(func=my_tool, description="Custom description")
        assert t.description == "Custom description"

    def test_description_no_docstring(self):
        def my_tool():
            pass

        t = Tool(func=my_tool)
        assert "my_tool" in t.description

    def test_schema_auto_generated(self):
        def calc(expression: str) -> str:
            """Calculate."""
            return ""

        t = Tool(func=calc)
        assert isinstance(t.schema, ToolSchema)
        assert t.schema.name == "calc"

    def test_schema_override(self):
        def calc(expression: str) -> str:
            return ""

        custom_schema = ToolSchema(name="calculator", description="Calc", parameters=[])
        t = Tool(func=calc, schema=custom_schema)
        assert t.schema.name == "calculator"

    def test_config_default(self):
        def fn():
            pass

        t = Tool(func=fn)
        assert isinstance(t.config, ToolConfig)
        assert t.config.timeout == 60.0

    def test_config_custom(self):
        def fn():
            pass

        cfg = ToolConfig(timeout=10.0, max_retries=2)
        t = Tool(func=fn, config=cfg)
        assert t.config.timeout == 10.0
        assert t.config.max_retries == 2

    def test_repr(self):
        def fn():
            pass

        t = Tool(func=fn, name="test_tool")
        assert repr(t) == "Tool(name='test_tool')"

    @pytest.mark.asyncio
    async def test_execute_sync(self):
        def add(a: int, b: int) -> int:
            """Add numbers.

            Args:
                a: First number
                b: Second number
            """
            return a + b

        t = Tool(func=add)
        result = await t.execute(a=2, b=3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_async(self):
        async def greet(name: str) -> str:
            """Greet someone.

            Args:
                name: The name
            """
            return f"Hello, {name}!"

        t = Tool(func=greet)
        result = await t.execute(name="Alice")
        assert result == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self):
        def concat(a: str, b: str, sep: str = " ") -> str:
            """Concat strings.

            Args:
                a: First string
                b: Second string
                sep: Separator
            """
            return f"{a}{sep}{b}"

        t = Tool(func=concat)
        result = await t.execute(a="hello", b="world")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        def bad_tool(x: int) -> int:
            """Fail.

            Args:
                x: Input
            """
            raise ValueError("bad input")

        t = Tool(func=bad_tool)
        with pytest.raises(ToolExecutionError) as exc_info:
            await t.execute(x=1)
        assert "bad_tool" in str(exc_info.value)
        assert "bad input" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        async def slow_tool(x: int) -> int:
            """Slow.

            Args:
                x: Input
            """
            await asyncio.sleep(10)
            return x

        t = Tool(func=slow_tool, config=ToolConfig(timeout=0.05))
        with pytest.raises(ToolTimeoutError) as exc_info:
            await t.execute(x=1)
        assert "slow_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_retry_on_failure(self):
        call_count = 0

        def flaky_tool(x: int) -> int:
            """Flaky.

            Args:
                x: Input
            """
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return x * 2

        t = Tool(func=flaky_tool, config=ToolConfig(max_retries=2, retry_backoff=0.01))
        result = await t.execute(x=5)
        assert result == 10
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_retry_exhausted(self):
        def always_fail(x: int) -> int:
            """Fail.

            Args:
                x: Input
            """
            raise RuntimeError("permanent error")

        t = Tool(func=always_fail, config=ToolConfig(max_retries=1, retry_backoff=0.01))
        with pytest.raises(ToolExecutionError):
            await t.execute(x=1)

    def test_to_llm_schema(self):
        def calc(expression: str) -> str:
            """Calculate a math expression.

            Args:
                expression: The expression to evaluate
            """
            return ""

        t = Tool(func=calc)
        llm_schema = t.to_llm_schema()
        assert llm_schema.name == "calc"
        assert llm_schema.description == "Calculate a math expression."

    def test_with_type_hints(self):
        def fn(name: str, count: int, active: bool = True) -> str:
            """Do something.

            Args:
                name: The name
                count: The count
                active: Whether active
            """
            return ""

        t = Tool(func=fn)
        param_names = [p.name for p in t.schema.parameters]
        assert "name" in param_names
        assert "count" in param_names
        assert "active" in param_names

    def test_with_default_args(self):
        def fn(required_arg: str, optional_arg: int = 42) -> str:
            """Fn.

            Args:
                required_arg: Required
                optional_arg: Optional
            """
            return ""

        t = Tool(func=fn)
        param_map = {p.name: p for p in t.schema.parameters}
        assert param_map["required_arg"].required is True
        assert param_map["optional_arg"].required is False


# ===================================================================
# @tool decorator
# ===================================================================


class TestToolDecorator:
    def test_basic_no_args(self):
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        assert isinstance(search, Tool)
        assert search.name == "search"
        assert search.description == "Search the web."

    def test_with_name(self):
        @tool(name="web_search")
        def search(query: str) -> str:
            """Search."""
            return "results"

        assert isinstance(search, Tool)
        assert search.name == "web_search"

    def test_with_description(self):
        @tool(description="Custom description")
        def search(query: str) -> str:
            """Original."""
            return "results"

        assert isinstance(search, Tool)
        assert search.description == "Custom description"

    def test_with_timeout(self):
        @tool(timeout=30.0)
        def search(query: str) -> str:
            """Search."""
            return "results"

        assert search.config.timeout == 30.0

    def test_with_retries(self):
        @tool(retries=3)
        def search(query: str) -> str:
            """Search."""
            return "results"

        assert search.config.max_retries == 3

    def test_with_require_confirmation(self):
        @tool(require_confirmation=True)
        def danger(cmd: str) -> str:
            """Dangerous."""
            return ""

        assert danger.config.require_confirmation is True

    def test_with_all_args(self):
        @tool(name="custom", description="Custom tool", timeout=15.0, retries=2, require_confirmation=True)
        def fn(x: int) -> int:
            """Original."""
            return x

        assert fn.name == "custom"
        assert fn.description == "Custom tool"
        assert fn.config.timeout == 15.0
        assert fn.config.max_retries == 2
        assert fn.config.require_confirmation is True

    @pytest.mark.asyncio
    async def test_decorated_tool_is_executable(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add.

            Args:
                a: First
                b: Second
            """
            return a + b

        result = await add.execute(a=3, b=4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_decorated_async_tool(self):
        @tool
        async def fetch(url: str) -> str:
            """Fetch.

            Args:
                url: URL to fetch
            """
            return f"content from {url}"

        result = await fetch.execute(url="http://example.com")
        assert result == "content from http://example.com"

    def test_decorator_preserves_name(self):
        @tool
        def my_func(x: int) -> int:
            """My func."""
            return x

        assert my_func.name == "my_func"
