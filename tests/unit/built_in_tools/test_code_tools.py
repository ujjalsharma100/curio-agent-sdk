"""
Unit tests for built-in code execution tools (python_execute, shell_execute).
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.tools.code import python_execute, shell_execute


@pytest.mark.unit
class TestPythonExecute:
    def test_python_execute_simple(self):
        result = python_execute.func("print(2 + 2)")
        assert "4" in result

    def test_python_execute_output(self):
        result = python_execute.func("print('hello'); print('world')")
        assert "hello" in result
        assert "world" in result

    def test_python_execute_error(self):
        result = python_execute.func("1/0")
        assert "Error" in result or "ZeroDivisionError" in result or "STDERR" in result
        result_syntax = python_execute.func("syntax error [")
        assert "Error" in result_syntax or "SyntaxError" in result_syntax or "STDERR" in result_syntax

    def test_python_execute_is_tool(self):
        assert isinstance(python_execute, Tool)
        assert python_execute.name == "python_execute"


@pytest.mark.unit
class TestShellExecute:
    def test_shell_execute_simple(self):
        result = shell_execute.func("echo hello")
        assert "hello" in result

    def test_shell_execute_error(self):
        result = shell_execute.func("exit 42")
        assert "42" in result or "Exit code" in result or "(no output)" in result

    def test_shell_execute_timeout(self):
        with patch("curio_agent_sdk.tools.code._sandboxed_run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 25)
            result = shell_execute.func("sleep 99")
            assert "timed out" in result or "Error" in result

    def test_shell_execute_is_tool(self):
        assert isinstance(shell_execute, Tool)
        assert shell_execute.name == "shell_execute"
