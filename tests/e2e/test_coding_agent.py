"""
E2E tests: Coding Agent (Phase 18 §22.4)

Validates agent reading/writing files and executing code via tools.
"""

import os
import tempfile
import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    with open(path, "r") as f:
        return f.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to {path}"


@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    import io
    import contextlib

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__})
        return output.getvalue().strip() or "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_file_read_write_agent():
    """Agent reads and writes files as part of a coding task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "hello.py")

        mock = MockLLM()
        # Step 1: Write a file
        mock.add_tool_call_response(
            "write_file",
            {"path": test_file, "content": "print('Hello, World!')"},
        )
        # Step 2: Read the file back
        mock.add_tool_call_response("read_file", {"path": test_file})
        # Step 3: Final response
        mock.add_text_response("I've created hello.py with a Hello World program and verified its contents.")

        agent = Agent(
            system_prompt="You are a coding assistant. Create and verify files.",
            tools=[read_file, write_file],
            llm=mock,
        )
        harness = AgentTestHarness(agent, llm=mock)
        result = await harness.run("Create a hello world Python file and verify it")

        assert result.status == "completed"
        assert len(harness.tool_calls) == 2
        assert harness.tool_calls[0][0] == "write_file"
        assert harness.tool_calls[1][0] == "read_file"

        # Verify the file was actually written
        assert os.path.exists(test_file)
        with open(test_file) as f:
            content = f.read()
        assert "Hello, World!" in content


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_code_execution_agent():
    """Agent executes Python code and uses the output."""
    mock = MockLLM()
    mock.add_tool_call_response(
        "run_python",
        {"code": "result = sum(range(1, 11))\nprint(result)"},
    )
    mock.add_text_response("The sum of numbers 1 through 10 is 55.")

    agent = Agent(
        system_prompt="You are a coding assistant. Use the run_python tool to execute code.",
        tools=[run_python],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Calculate the sum of 1 to 10")

    assert result.status == "completed"
    assert len(harness.tool_calls) == 1
    assert harness.tool_calls[0][0] == "run_python"
    assert "55" in result.output
