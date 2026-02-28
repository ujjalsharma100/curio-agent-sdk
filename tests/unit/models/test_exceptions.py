"""
Unit tests for curio_agent_sdk.models.exceptions

Covers: Full exception hierarchy, attributes, inheritance chains, messages
"""

import pytest

from curio_agent_sdk.models.exceptions import (
    AgentCancelledError,
    AgentError,
    AgentTimeoutError,
    ConfigError,
    CostBudgetExceeded,
    CurioError,
    LLMAuthenticationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    MaxIterationsError,
    NoAvailableModelError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)


# ===================================================================
# Base
# ===================================================================


class TestCurioError:
    def test_is_exception(self):
        assert issubclass(CurioError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(CurioError):
            raise CurioError("test")

    def test_message_preserved(self):
        err = CurioError("something went wrong")
        assert str(err) == "something went wrong"


# ===================================================================
# LLM Errors
# ===================================================================


class TestLLMError:
    def test_hierarchy(self):
        assert issubclass(LLMError, CurioError)

    def test_creation(self):
        err = LLMError("fail", provider="openai", model="gpt-4o")
        assert err.provider == "openai"
        assert err.model == "gpt-4o"
        assert str(err) == "fail"

    def test_defaults(self):
        err = LLMError("fail")
        assert err.provider == ""
        assert err.model == ""

    def test_catchable_as_curio_error(self):
        with pytest.raises(CurioError):
            raise LLMError("test")


class TestLLMRateLimitError:
    def test_hierarchy(self):
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMRateLimitError, CurioError)

    def test_creation(self):
        err = LLMRateLimitError(provider="openai", model="gpt-4o", retry_after=30.0)
        assert err.provider == "openai"
        assert err.model == "gpt-4o"
        assert err.retry_after == 30.0
        assert "Rate limit exceeded" in str(err)
        assert "retry after 30.0s" in str(err)

    def test_without_retry_after(self):
        err = LLMRateLimitError(provider="openai", model="gpt-4o")
        assert err.retry_after is None
        assert "retry after" not in str(err)

    def test_catchable_as_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMRateLimitError("openai", "gpt-4o")


class TestLLMAuthenticationError:
    def test_hierarchy(self):
        assert issubclass(LLMAuthenticationError, LLMError)
        assert issubclass(LLMAuthenticationError, CurioError)

    def test_creation(self):
        err = LLMAuthenticationError("Invalid API key", provider="openai")
        assert err.provider == "openai"


class TestLLMProviderError:
    def test_hierarchy(self):
        assert issubclass(LLMProviderError, LLMError)

    def test_creation(self):
        err = LLMProviderError(
            "Internal server error",
            provider="anthropic",
            model="claude-sonnet",
            status_code=500,
        )
        assert err.status_code == 500
        assert err.provider == "anthropic"
        assert err.model == "claude-sonnet"

    def test_without_status_code(self):
        err = LLMProviderError("Network error")
        assert err.status_code is None


class TestLLMTimeoutError:
    def test_hierarchy(self):
        assert issubclass(LLMTimeoutError, LLMError)

    def test_creation(self):
        err = LLMTimeoutError("Timed out", provider="openai", model="gpt-4o")
        assert str(err) == "Timed out"


class TestNoAvailableModelError:
    def test_hierarchy(self):
        assert issubclass(NoAvailableModelError, LLMError)

    def test_creation(self):
        err = NoAvailableModelError("No models in tier1")
        assert "No models" in str(err)


class TestCostBudgetExceeded:
    def test_hierarchy(self):
        assert issubclass(CostBudgetExceeded, LLMError)

    def test_creation(self):
        err = CostBudgetExceeded(total_cost=1.5, budget=1.0)
        assert err.total_cost == 1.5
        assert err.budget == 1.0
        assert "$1.5000" in str(err)
        assert "$1.0000" in str(err)
        assert "exceeded" in str(err).lower()


# ===================================================================
# Tool Errors
# ===================================================================


class TestToolError:
    def test_hierarchy(self):
        assert issubclass(ToolError, CurioError)

    def test_creation(self):
        err = ToolError("fail", tool_name="calculator")
        assert err.tool_name == "calculator"
        assert str(err) == "fail"

    def test_defaults(self):
        err = ToolError("fail")
        assert err.tool_name == ""

    def test_catchable_as_curio_error(self):
        with pytest.raises(CurioError):
            raise ToolError("test")


class TestToolNotFoundError:
    def test_hierarchy(self):
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolNotFoundError, CurioError)

    def test_creation(self):
        err = ToolNotFoundError("search", available=["calc", "web"])
        assert err.tool_name == "search"
        assert err.available == ["calc", "web"]
        assert "'search' not found" in str(err)

    def test_without_available(self):
        err = ToolNotFoundError("search")
        assert err.available == []


class TestToolExecutionError:
    def test_hierarchy(self):
        assert issubclass(ToolExecutionError, ToolError)

    def test_creation(self):
        cause = ValueError("bad input")
        err = ToolExecutionError("calc", cause)
        assert err.tool_name == "calc"
        assert err.cause is cause
        assert "calc" in str(err)
        assert "bad input" in str(err)


class TestToolTimeoutError:
    def test_hierarchy(self):
        assert issubclass(ToolTimeoutError, ToolError)

    def test_creation(self):
        err = ToolTimeoutError("slow_tool", timeout=30.0)
        assert err.tool_name == "slow_tool"
        assert err.timeout == 30.0
        assert "30" in str(err)
        assert "slow_tool" in str(err)


class TestToolValidationError:
    def test_hierarchy(self):
        assert issubclass(ToolValidationError, ToolError)

    def test_creation(self):
        errors = ["missing 'query'", "invalid type for 'limit'"]
        err = ToolValidationError("search", errors=errors)
        assert err.tool_name == "search"
        assert err.errors == errors
        assert "missing 'query'" in str(err)
        assert "invalid type" in str(err)

    def test_single_error(self):
        err = ToolValidationError("fn", errors=["required field missing"])
        assert len(err.errors) == 1


# ===================================================================
# Agent Errors
# ===================================================================


class TestAgentError:
    def test_hierarchy(self):
        assert issubclass(AgentError, CurioError)

    def test_creation(self):
        err = AgentError("agent fail")
        assert str(err) == "agent fail"


class TestAgentTimeoutError:
    def test_hierarchy(self):
        assert issubclass(AgentTimeoutError, AgentError)
        assert issubclass(AgentTimeoutError, CurioError)

    def test_creation(self):
        err = AgentTimeoutError(timeout=60.0, iterations_completed=3)
        assert err.timeout == 60.0
        assert err.iterations_completed == 3
        assert "60" in str(err)
        assert "3" in str(err)


class TestAgentCancelledError:
    def test_hierarchy(self):
        assert issubclass(AgentCancelledError, AgentError)

    def test_creation(self):
        err = AgentCancelledError("Run was cancelled")
        assert str(err) == "Run was cancelled"


class TestMaxIterationsError:
    def test_hierarchy(self):
        assert issubclass(MaxIterationsError, AgentError)

    def test_creation(self):
        err = MaxIterationsError(max_iterations=25)
        assert err.max_iterations == 25
        assert "25" in str(err)


# ===================================================================
# Config Errors
# ===================================================================


class TestConfigError:
    def test_hierarchy(self):
        assert issubclass(ConfigError, CurioError)

    def test_creation(self):
        err = ConfigError("Bad config")
        assert str(err) == "Bad config"


# ===================================================================
# Cross-cutting: catch-all with CurioError
# ===================================================================


class TestCatchAll:
    @pytest.mark.parametrize(
        "exc_class,args",
        [
            (LLMError, ("msg",)),
            (LLMRateLimitError, ("openai", "gpt-4o")),
            (LLMAuthenticationError, ("msg",)),
            (LLMProviderError, ("msg",)),
            (LLMTimeoutError, ("msg",)),
            (NoAvailableModelError, ("msg",)),
            (CostBudgetExceeded, (1.0, 0.5)),
            (ToolError, ("msg",)),
            (ToolNotFoundError, ("tool",)),
            (ToolExecutionError, ("tool", ValueError("x"))),
            (ToolTimeoutError, ("tool", 10.0)),
            (ToolValidationError, ("tool", ["err"])),
            (AgentError, ("msg",)),
            (AgentTimeoutError, (60.0, 5)),
            (AgentCancelledError, ("msg",)),
            (MaxIterationsError, (25,)),
            (ConfigError, ("msg",)),
        ],
    )
    def test_all_catchable_by_curio_error(self, exc_class, args):
        with pytest.raises(CurioError):
            raise exc_class(*args)

    @pytest.mark.parametrize(
        "exc_class,args",
        [
            (LLMRateLimitError, ("openai", "gpt-4o")),
            (LLMAuthenticationError, ("msg",)),
            (LLMProviderError, ("msg",)),
            (LLMTimeoutError, ("msg",)),
            (NoAvailableModelError, ("msg",)),
            (CostBudgetExceeded, (1.0, 0.5)),
        ],
    )
    def test_llm_errors_catchable_by_llm_error(self, exc_class, args):
        with pytest.raises(LLMError):
            raise exc_class(*args)

    @pytest.mark.parametrize(
        "exc_class,args",
        [
            (ToolNotFoundError, ("tool",)),
            (ToolExecutionError, ("tool", ValueError("x"))),
            (ToolTimeoutError, ("tool", 10.0)),
            (ToolValidationError, ("tool", ["err"])),
        ],
    )
    def test_tool_errors_catchable_by_tool_error(self, exc_class, args):
        with pytest.raises(ToolError):
            raise exc_class(*args)

    @pytest.mark.parametrize(
        "exc_class,args",
        [
            (AgentTimeoutError, (60.0, 5)),
            (AgentCancelledError, ("msg",)),
            (MaxIterationsError, (25,)),
        ],
    )
    def test_agent_errors_catchable_by_agent_error(self, exc_class, args):
        with pytest.raises(AgentError):
            raise exc_class(*args)
