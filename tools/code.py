"""Built-in code execution tools."""

import subprocess
import sys
import tempfile

from curio_agent_sdk.core.tools.tool import tool


@tool(
    name="python_execute",
    description="Execute Python code and return stdout/stderr output",
    timeout=30.0,
    require_confirmation=True,
)
def python_execute(code: str) -> str:
    """Execute Python code in a subprocess. Returns stdout and stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=25,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: execution timed out after 25 seconds"
        except Exception as e:
            return f"Error: {e}"


@tool(
    name="shell_execute",
    description="Execute a shell command and return its output",
    timeout=30.0,
    require_confirmation=True,
)
def shell_execute(command: str) -> str:
    """Execute a shell command. Returns stdout and stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=25,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 25 seconds"
    except Exception as e:
        return f"Error: {e}"
