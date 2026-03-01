"""
Unit tests for HumanInputHandler and implementations (CLI, mock).
"""

import pytest
from unittest.mock import patch

from curio_agent_sdk.core.security.human_input import (
    HumanInputHandler,
    CLIHumanInput,
)


class MockHumanInputApprove(HumanInputHandler):
    """Concrete handler that always approves."""

    async def confirm_tool_call(self, tool_name: str, args: dict) -> bool:
        return True

    async def get_input(self, prompt: str) -> str:
        return "approved"


class MockHumanInputDeny(HumanInputHandler):
    """Concrete handler that always denies."""

    async def confirm_tool_call(self, tool_name: str, args: dict) -> bool:
        return False

    async def get_input(self, prompt: str) -> str:
        return "denied"


@pytest.mark.unit
class TestHumanInputHandler:
    def test_human_input_handler_is_abstract(self):
        """HumanInputHandler cannot be instantiated (abstract)."""
        with pytest.raises(TypeError):
            HumanInputHandler()  # type: ignore[abstract-instance]


@pytest.mark.unit
class TestMockHumanInput:
    @pytest.mark.asyncio
    async def test_mock_human_input_approve(self):
        handler = MockHumanInputApprove()
        result = await handler.confirm_tool_call("run_shell", {"cmd": "ls"})
        assert result is True
        text = await handler.get_input("Say something")
        assert text == "approved"

    @pytest.mark.asyncio
    async def test_mock_human_input_deny(self):
        handler = MockHumanInputDeny()
        result = await handler.confirm_tool_call("run_shell", {"cmd": "rm -rf /"})
        assert result is False
        text = await handler.get_input("Say something")
        assert text == "denied"


@pytest.mark.unit
class TestCLIHumanInput:
    @pytest.mark.asyncio
    async def test_cli_human_input_approve(self):
        handler = CLIHumanInput()
        with patch("builtins.input", return_value="y"):
            result = await handler.confirm_tool_call("test_tool", {"arg": 1})
        assert result is True

    @pytest.mark.asyncio
    async def test_cli_human_input_deny(self):
        handler = CLIHumanInput()
        with patch("builtins.input", return_value="n"):
            result = await handler.confirm_tool_call("test_tool", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_cli_human_input_yes_approves(self):
        handler = CLIHumanInput()
        with patch("builtins.input", return_value="yes"):
            result = await handler.confirm_tool_call("tool", {})
        assert result is True

    @pytest.mark.asyncio
    async def test_cli_human_input_empty_denies(self):
        handler = CLIHumanInput()
        with patch("builtins.input", return_value=""):
            result = await handler.confirm_tool_call("tool", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_cli_get_input(self):
        handler = CLIHumanInput()
        with patch("builtins.input", return_value="  hello world  "):
            out = await handler.get_input("Prompt:")
        assert out == "hello world"
