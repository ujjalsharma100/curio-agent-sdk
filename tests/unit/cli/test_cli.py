"""
Unit tests for curio_agent_sdk.cli (Phase 15 — CLI).

Covers: AgentCLI — creation, wrapping agent, command parsing, exit, help.
"""

import pytest

from curio_agent_sdk.cli import AgentCLI
from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.state import SessionManager, InMemorySessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_agent():
    """Minimal agent for CLI tests."""
    return Agent(agent_id="cli-test-agent", agent_name="CLITestAgent")


@pytest.fixture
def cli(sample_agent):
    """AgentCLI instance wrapping sample_agent."""
    return AgentCLI(sample_agent)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_creation(sample_agent):
    """AgentCLI constructor sets agent and session manager."""
    cli = AgentCLI(sample_agent)
    assert cli.agent is sample_agent
    assert cli.session_manager is not None
    assert cli.current_session_id is None
    assert cli._should_exit is False
    # Built-in commands are registered
    assert "/help" in cli._commands
    assert "/exit" in cli._commands


@pytest.mark.unit
def test_cli_with_agent(sample_agent):
    """CLI wraps the provided Agent and agent gets session_manager."""
    cli = AgentCLI(sample_agent)
    assert cli.agent is sample_agent
    assert sample_agent.session_manager is cli.session_manager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_command_parsing(cli, capsys):
    """Slash commands are parsed and dispatched to registered handlers."""
    # /help is a built-in; running it should print help text
    await cli._run_command("/help")
    captured = capsys.readouterr()
    assert "Available commands" in captured.out
    assert "/help" in captured.out
    assert "/exit" in captured.out
    # Unknown command prints message
    await cli._run_command("/unknown_cmd")
    captured2 = capsys.readouterr()
    assert "Unknown command" in captured2.out
    assert "/unknown_cmd" in captured2.out


@pytest.mark.unit
def test_cli_exit_command(cli):
    """Exit command sets _should_exit so REPL can exit."""
    assert cli._should_exit is False
    cli._cmd_exit("")
    assert cli._should_exit is True


@pytest.mark.unit
def test_cli_help_command(cli, capsys):
    """Help command lists available slash commands."""
    cli._cmd_help("")
    captured = capsys.readouterr()
    assert "Available commands" in captured.out
    assert "/help" in captured.out
    assert "/exit" in captured.out
    assert "Type a message to send it to the agent" in captured.out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_register_command(cli, capsys):
    """Custom slash commands can be registered and invoked."""
    seen = []

    def my_handler(args: str):
        seen.append(args)

    cli.register_command("ping", my_handler, "Reply with pong")
    assert "/ping" in cli._commands
    await cli._run_command("/ping")
    assert seen == [""]
    await cli._run_command("/ping extra")
    assert seen == ["", "extra"]
    # Name without leading slash is normalized
    cli.register_command("pong", my_handler)
    assert "/pong" in cli._commands


@pytest.mark.unit
def test_cli_register_command_empty_name_raises(cli):
    """Registering a command with empty name raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        cli.register_command("", lambda a: None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_run_once(sample_agent, capsys):
    """run_once runs agent once and prints result; returns exit code."""
    from unittest.mock import AsyncMock
    from curio_agent_sdk.models.agent import AgentRunResult

    result = AgentRunResult(
        status="completed",
        output="Hello back",
        total_iterations=1,
        total_llm_calls=1,
        total_tool_calls=0,
        total_input_tokens=5,
        total_output_tokens=10,
        run_id="run-1",
    )
    sample_agent.arun = AsyncMock(return_value=result)
    cli = AgentCLI(sample_agent)

    code = await cli.run_once("Hi", stream=False)
    assert code == 0
    captured = capsys.readouterr()
    assert "COMPLETED" in captured.out
    assert "Hello back" in captured.out

    result_fail = AgentRunResult(
        status="failed",
        output="",
        total_iterations=0,
        total_llm_calls=0,
        total_tool_calls=0,
        total_input_tokens=0,
        total_output_tokens=0,
        run_id="run-2",
    )
    sample_agent.arun = AsyncMock(return_value=result_fail)
    code_fail = await cli.run_once("Hi", stream=False)
    assert code_fail == 1


@pytest.mark.unit
def test_cli_cmd_status(cli, capsys):
    """Status command prints agent name, id, and session info."""
    cli._cmd_status("")
    captured = capsys.readouterr()
    assert "CLITestAgent" in captured.out
    assert "cli-test-agent" in captured.out
    assert "Current session" in captured.out
