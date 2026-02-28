"""
Tool class and @tool decorator for defining agent tools.

Tools wrap functions with schema, configuration, and execution behavior.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from curio_agent_sdk.core.tools.schema import ToolSchema


@dataclass
class ToolConfig:
    """Configuration for tool execution behavior."""
    timeout: float = 60.0
    max_retries: int = 0
    retry_backoff: float = 1.0
    require_confirmation: bool = False
    cache_ttl: float | None = None  # TTL in seconds for caching results; None = no caching
    sandboxed: bool = False  # Flag for sandboxed execution (actual sandboxing deferred)
    idempotent: bool = False  # If True, retried calls with same args return previous result


class Tool:
    """
    A tool that an agent can use.

    Wraps a function with:
    - JSON Schema generation from type hints
    - Async execution with timeout and retry
    - Input validation
    - Configurable behavior
    """

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        schema: ToolSchema | None = None,
        config: ToolConfig | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").split("\n")[0].strip() or f"Execute {self.name}"
        self.schema = schema or ToolSchema.from_function(func, name=self.name, description=self.description)
        self.config = config or ToolConfig()
        self._is_async = asyncio.iscoroutinefunction(func)

    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with validated arguments.

        Handles:
        - Input validation against schema
        - Async/sync function dispatch
        - Timeout enforcement
        - Retry on failure
        """
        validated = self.schema.validate(kwargs)
        return await self._execute_with_retry(validated)

    async def _execute_with_retry(self, args: dict[str, Any]) -> Any:
        """Execute with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._run(args),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                from curio_agent_sdk.models.exceptions import ToolTimeoutError
                last_error = ToolTimeoutError(self.name, self.config.timeout)
                if attempt == self.config.max_retries:
                    raise last_error
            except Exception as e:
                last_error = e
                if attempt == self.config.max_retries:
                    from curio_agent_sdk.models.exceptions import ToolExecutionError
                    raise ToolExecutionError(self.name, e)

            # Exponential backoff before retry
            await asyncio.sleep(self.config.retry_backoff ** attempt)

        # Should not reach here, but safety net
        raise last_error  # type: ignore

    async def _run(self, args: dict[str, Any]) -> Any:
        """Execute the function (async or sync)."""
        if self._is_async:
            return await self.func(**args)
        else:
            return await asyncio.to_thread(self.func, **args)

    def to_llm_schema(self):
        """Get the LLM-compatible tool schema."""
        return self.schema.to_llm_schema()

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 60.0,
    retries: int = 0,
    require_confirmation: bool = False,
) -> Tool | Callable:
    """
    Decorator to create a Tool from a function.

    Can be used with or without arguments:

        @tool
        def search(query: str) -> str:
            '''Search the web.'''
            return results

        @tool(timeout=30, retries=2)
        def fetch_data(url: str) -> dict:
            '''Fetch data from a URL.'''
            return data

        @tool(name="calculator", description="Evaluate math expressions")
        def calc(expression: str) -> float:
            return eval(expression)
    """
    def decorator(fn: Callable) -> Tool:
        return Tool(
            func=fn,
            name=name or fn.__name__,
            description=description,
            config=ToolConfig(
                timeout=timeout,
                max_retries=retries,
                require_confirmation=require_confirmation,
            ),
        )

    if func is not None:
        # @tool without arguments
        return decorator(func)

    # @tool(...) with arguments
    return decorator
