"""Built-in file I/O tools."""

import os

from curio_agent_sdk.core.tools.tool import tool


@tool(name="file_read", description="Read a file and return its contents", timeout=10.0)
def file_read(path: str) -> str:
    """Read a file and return its text content. Truncated to 50000 chars."""
    try:
        path = os.path.expanduser(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 50000:
            content = content[:50000] + "\n... [truncated]"
        return content
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool(
    name="file_write",
    description="Write content to a file",
    timeout=10.0,
    require_confirmation=True,
)
def file_write(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"
