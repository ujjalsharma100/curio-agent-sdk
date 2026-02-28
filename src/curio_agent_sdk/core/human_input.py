"""
Human-in-the-loop input handlers.

Provides an abstract base class and a CLI implementation for
getting human confirmation before tool execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class HumanInputHandler(ABC):
    """
    Abstract base class for human-in-the-loop input.

    Implement this to control how the agent requests confirmation
    or input from a human operator.

    Example:
        class SlackHumanInput(HumanInputHandler):
            async def confirm_tool_call(self, tool_name, args):
                # Send Slack message and wait for response
                ...

            async def get_input(self, prompt):
                # Send Slack message and wait for response
                ...
    """

    @abstractmethod
    async def confirm_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        """
        Ask the human to confirm a tool call.

        Args:
            tool_name: Name of the tool to execute.
            args: Arguments that will be passed to the tool.

        Returns:
            True if the human approves, False to deny.
        """
        ...

    @abstractmethod
    async def get_input(self, prompt: str) -> str:
        """
        Get free-form input from the human.

        Args:
            prompt: The prompt to display.

        Returns:
            The human's input string.
        """
        ...


class CLIHumanInput(HumanInputHandler):
    """
    Terminal-based human input handler.

    Prompts the user via stdin/stdout for tool call confirmation
    and free-form input.

    Example:
        agent = Agent(
            human_input=CLIHumanInput(),
            tools=[tool_with_confirmation],
            ...
        )
    """

    async def confirm_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Prompt the user in the terminal to confirm a tool call."""
        args_str = json.dumps(args, indent=2, default=str)
        prompt = (
            f"\n--- Tool Confirmation Required ---\n"
            f"Tool: {tool_name}\n"
            f"Args: {args_str}\n"
            f"Allow this tool call? [y/N]: "
        )

        # Run input() in a thread to avoid blocking the event loop
        response = await asyncio.to_thread(input, prompt)
        approved = response.strip().lower() in ("y", "yes")

        if not approved:
            logger.info("User denied tool call: %s", tool_name)

        return approved

    async def get_input(self, prompt: str) -> str:
        """Prompt the user for free-form input in the terminal."""
        response = await asyncio.to_thread(input, f"\n{prompt}: ")
        return response.strip()
