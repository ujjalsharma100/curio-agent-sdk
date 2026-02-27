"""Built-in code execution tools."""

import os
import subprocess
import sys
import tempfile
from typing import Mapping

from curio_agent_sdk.core.tools.tool import tool


def _sandboxed_run(
    args: list[str] | str,
    *,
    shell: bool = False,
    timeout: int = 25,
    env_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run a subprocess with basic OS-level sandboxing.

    The sandbox is intentionally conservative:
    - Runs in a restricted working directory (CURIO_SANDBOX_CWD or CWD)
    - Uses a minimal environment (PATH, LANG, LC_ALL plus explicit overrides)
    - On POSIX, applies simple resource limits (CPU time and address space)
    """
    cwd = os.getenv("CURIO_SANDBOX_CWD") or os.getcwd()

    base_env: dict[str, str] = {}
    for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE"):
        if key in os.environ:
            base_env[key] = os.environ[key]
    if env_overrides:
        base_env.update({k: str(v) for k, v in env_overrides.items()})

    kwargs: dict[str, object] = {
        "cwd": cwd,
        "env": base_env,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }

    if shell:
        kwargs["shell"] = True

    if os.name == "posix":
        # Apply simple rlimits for CPU time and virtual memory.
        def _preexec() -> None:  # pragma: no cover - depends on OS
            try:
                import resource

                # Max 5 seconds of CPU time
                resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
                # Limit address space to ~512MB
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (512 * 1024 * 1024, 512 * 1024 * 1024),
                )
            except Exception:
                # Best-effort; failure to apply limits should not crash the tool.
                pass

        kwargs["preexec_fn"] = _preexec  # type: ignore[assignment]

    return subprocess.run(args, **kwargs)  # type: ignore[arg-type]


@tool(
    name="python_execute",
    description="Execute Python code and return stdout/stderr output (sandboxed subprocess)",
    timeout=30.0,
    require_confirmation=True,
)
def python_execute(code: str) -> str:
    """Execute Python code in a sandboxed subprocess. Returns stdout and stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = _sandboxed_run(
                [sys.executable, f.name],
                shell=False,
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
    description="Execute a shell command in a sandboxed subprocess and return its output",
    timeout=30.0,
    require_confirmation=True,
)
def shell_execute(command: str) -> str:
    """Execute a shell command in a sandboxed subprocess. Returns stdout and stderr."""
    try:
        result = _sandboxed_run(
            command,
            shell=True,
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
