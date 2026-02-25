"""
Custom exception hierarchy for the Curio Agent SDK.
"""


class CurioError(Exception):
    """Base exception for all Curio Agent SDK errors."""
    pass


# === LLM Errors ===

class LLMError(CurioError):
    """Base exception for LLM-related errors."""
    def __init__(self, message: str, provider: str = "", model: str = ""):
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Rate limit exceeded for an LLM provider."""
    def __init__(self, provider: str, model: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {provider}/{model}"
            + (f" (retry after {retry_after}s)" if retry_after else ""),
            provider=provider,
            model=model,
        )


class LLMAuthenticationError(LLMError):
    """Authentication failed for an LLM provider."""
    pass


class LLMProviderError(LLMError):
    """Provider returned an error (5xx, network, etc.)."""
    def __init__(self, message: str, provider: str = "", model: str = "", status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message, provider=provider, model=model)


class LLMTimeoutError(LLMError):
    """LLM call timed out."""
    pass


class NoAvailableModelError(LLMError):
    """No model available in the requested tier after exhausting all options."""
    pass


class CostBudgetExceeded(LLMError):
    """Cost budget has been exceeded."""
    def __init__(self, total_cost: float, budget: float):
        self.total_cost = total_cost
        self.budget = budget
        super().__init__(f"Cost budget exceeded: ${total_cost:.4f} > ${budget:.4f}")


# === Tool Errors ===

class ToolError(CurioError):
    """Base exception for tool-related errors."""
    def __init__(self, message: str, tool_name: str = ""):
        self.tool_name = tool_name
        super().__init__(message)


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""
    def __init__(self, tool_name: str, available: list[str] | None = None):
        self.available = available or []
        super().__init__(
            f"Tool '{tool_name}' not found. Available: {self.available}",
            tool_name=tool_name,
        )


class ToolExecutionError(ToolError):
    """Tool execution failed."""
    def __init__(self, tool_name: str, cause: Exception):
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' failed: {cause}", tool_name=tool_name)


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""
    def __init__(self, tool_name: str, timeout: float):
        self.timeout = timeout
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout}s",
            tool_name=tool_name,
        )


class ToolValidationError(ToolError):
    """Tool argument validation failed."""
    def __init__(self, tool_name: str, errors: list[str]):
        self.errors = errors
        super().__init__(
            f"Validation failed for tool '{tool_name}': {'; '.join(errors)}",
            tool_name=tool_name,
        )


# === Agent Errors ===

class AgentError(CurioError):
    """Base exception for agent-related errors."""
    pass


class AgentTimeoutError(AgentError):
    """Agent run timed out."""
    def __init__(self, timeout: float, iterations_completed: int):
        self.timeout = timeout
        self.iterations_completed = iterations_completed
        super().__init__(
            f"Agent timed out after {timeout}s ({iterations_completed} iterations completed)"
        )


class AgentCancelledError(AgentError):
    """Agent run was cancelled."""
    pass


class MaxIterationsError(AgentError):
    """Agent reached maximum iteration limit."""
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        super().__init__(f"Agent reached max iterations: {max_iterations}")


# === Config Errors ===

class ConfigError(CurioError):
    """Configuration error."""
    pass
