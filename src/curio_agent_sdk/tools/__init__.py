"""
Built-in tools for the Curio Agent SDK.

Ready-to-use tools for common agent tasks.

Example:
    from curio_agent_sdk.tools import web_fetch, file_read, http_request

    agent = Agent(
        tools=[web_fetch, file_read, http_request],
        ...
    )
"""

from curio_agent_sdk.tools.web import web_fetch
from curio_agent_sdk.tools.code import python_execute, shell_execute
from curio_agent_sdk.tools.file import file_read, file_write
from curio_agent_sdk.tools.http import http_request

__all__ = [
    "web_fetch",
    "python_execute",
    "shell_execute",
    "file_read",
    "file_write",
    "http_request",
]
