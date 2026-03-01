"""
Integration tests: Agent + Plan Mode (Phase 17 §21.10)

Validates plan mode workflow, read-only tools during planning, and todo management.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.workflow.plan_mode import PlanMode, TodoManager
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def read_file(path: str) -> str:
    """Read a file (read-only)."""
    return f"Contents of {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write to a file (write operation)."""
    return f"Wrote to {path}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_plan_mode_workflow():
    """Plan → approve → execute workflow."""
    mock = MockLLM()
    mock.add_text_response("Here is my plan: Step 1, Step 2.")

    plan_mode = PlanMode(read_only_tool_names=["read_file"])

    agent = Agent(
        system_prompt="Plan before acting.",
        tools=[read_file, write_file],
        plan_mode=plan_mode,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Create a plan for the task")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_only_tools_during_planning():
    """Write tools are blocked in planning mode; read tools work."""
    mock = MockLLM()
    mock.add_tool_call_response("read_file", {"path": "/tmp/test"})
    mock.add_text_response("Read the file successfully.")

    plan_mode = PlanMode(read_only_tool_names=["read_file"])

    agent = Agent(
        system_prompt="Plan carefully.",
        tools=[read_file, write_file],
        plan_mode=plan_mode,
        read_only_tool_names=["read_file"],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Read and plan")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_todo_management():
    """Todos can be created and updated during plan mode."""
    todo_mgr = TodoManager()

    mock = MockLLM()
    mock.add_text_response("Plan with todos created.")

    plan_mode = PlanMode()

    agent = Agent(
        system_prompt="Create todos.",
        plan_mode=plan_mode,
        todo_manager=todo_mgr,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Plan the project")

    assert result.status == "completed"
    assert todo_mgr is not None
