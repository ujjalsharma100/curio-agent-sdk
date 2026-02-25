"""
Base class for agent execution loops.

Different loop patterns implement different agent architectures:
- ToolCallingLoop: Standard LLM tool calling (most common)
- PlanCritiqueSynthesizeLoop: Plan-execute-critique-synthesize
- Custom loops: Any pattern you need
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.models.events import StreamEvent


class AgentLoop(ABC):
    """
    Abstract base class for agent execution loops.

    A loop defines how an agent processes a request:
    - When to call the LLM
    - When to execute tools
    - When to stop
    - How to format the final output

    The Agent class drives the loop by calling step() repeatedly
    until should_continue() returns False or limits are reached.
    """

    @abstractmethod
    async def step(self, state: AgentState) -> AgentState:
        """
        Execute one step of the agent loop.

        A step typically involves:
        1. Calling the LLM with the current message history
        2. Processing the response (execute tools, update state)
        3. Returning the updated state

        Args:
            state: Current agent state (messages, metadata, etc.)

        Returns:
            Updated AgentState after this step.
        """
        ...

    @abstractmethod
    def should_continue(self, state: AgentState) -> bool:
        """
        Determine whether the loop should continue.

        Called after each step to decide if more iterations are needed.

        Args:
            state: Current agent state after the latest step.

        Returns:
            True if the loop should execute another step.
        """
        ...

    def get_output(self, state: AgentState) -> str:
        """
        Extract the final output from the agent state.

        Called after the loop completes to get the result.
        Default: returns the last assistant message's text content.

        Args:
            state: Final agent state.

        Returns:
            The agent's output string.
        """
        # Find the last assistant message
        for msg in reversed(state.messages):
            if msg.role == "assistant" and msg.content:
                text = msg.text if hasattr(msg, 'text') else str(msg.content)
                if text:
                    return text
        return ""

    async def stream_step(self, state: AgentState) -> AsyncIterator[StreamEvent]:
        """
        Stream a single step of the loop.

        Default implementation falls back to non-streaming step.
        Override for real streaming support.

        Args:
            state: Current agent state.

        Yields:
            StreamEvent objects for real-time observation.
        """
        updated = await self.step(state)
        # Yield the output as a single event
        output = self.get_output(updated)
        if output:
            yield StreamEvent(type="text_delta", text=output)
        yield StreamEvent(type="iteration_end", iteration=updated.iteration)
