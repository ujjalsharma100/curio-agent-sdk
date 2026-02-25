"""
Standard tool calling loop.

This is the most common agent pattern:
1. Send messages + tools to LLM
2. If LLM returns tool calls -> execute them -> add results -> loop
3. If LLM returns text (no tool calls) -> done

Used by: OpenAI Agents, Anthropic tool use, general-purpose agents.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.models.llm import LLMRequest, Message
from curio_agent_sdk.models.events import StreamEvent

logger = logging.getLogger(__name__)


class ToolCallingLoop(AgentLoop):
    """
    Standard tool calling loop using native LLM tool calling APIs.

    The LLM decides when to call tools and when it's done.
    This is the simplest and most widely-used agent pattern.

    Example:
        loop = ToolCallingLoop(llm=client, tool_executor=executor)
        # or let the Agent wire these up automatically
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tool_executor: ToolExecutor | None = None,
        tier: str = "tier2",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        run_id: str | None = None,
        agent_id: str | None = None,
    ):
        self.llm = llm
        self.tool_executor = tool_executor
        self.tier = tier
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.run_id = run_id
        self.agent_id = agent_id

    async def step(self, state: AgentState) -> AgentState:
        """
        One step: call LLM, optionally execute tools.
        """
        if not self.llm:
            raise RuntimeError("LLMClient not set on ToolCallingLoop")

        # Build request
        request = LLMRequest(
            messages=state.messages,
            tools=state.tool_schemas if state.tools else None,
            tool_choice="auto" if state.tool_schemas else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tier=self.tier,
        )

        # Call LLM
        response = await self.llm.call(request, run_id=self.run_id, agent_id=self.agent_id)
        state.record_llm_call(response.usage.input_tokens, response.usage.output_tokens)
        state._last_finish_reason = response.finish_reason

        # Add assistant message to history
        state.add_message(response.message)

        # If the LLM made tool calls, execute them
        if response.has_tool_calls and self.tool_executor:
            state.record_tool_calls(len(response.tool_calls))

            logger.info(f"Executing {len(response.tool_calls)} tool call(s): "
                       f"{[tc.name for tc in response.tool_calls]}")

            result_messages = await self.tool_executor.execute_to_messages(response.tool_calls)
            state.add_messages(result_messages)

        state.iteration += 1
        return state

    def should_continue(self, state: AgentState) -> bool:
        """Continue if the last response had tool calls (more work to do)."""
        if state.is_cancelled or state.is_done:
            return False
        if state.iteration >= state.max_iterations:
            return False
        # Continue only if the LLM asked to use tools
        return state._last_finish_reason == "tool_use"

    def get_output(self, state: AgentState) -> str:
        """Get the final text output from the last assistant message."""
        for msg in reversed(state.messages):
            if msg.role == "assistant":
                text = msg.text
                if text:
                    return text
        return ""

    async def stream_step(self, state: AgentState) -> AsyncIterator[StreamEvent]:
        """Stream a single step with real-time text and tool call events."""
        if not self.llm:
            raise RuntimeError("LLMClient not set on ToolCallingLoop")

        request = LLMRequest(
            messages=state.messages,
            tools=state.tool_schemas if state.tools else None,
            tool_choice="auto" if state.tool_schemas else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tier=self.tier,
        )

        # Accumulate the full response for state tracking
        full_text = ""
        tool_calls_completed = []

        async for chunk in self.llm.stream(request, run_id=self.run_id, agent_id=self.agent_id):
            if chunk.type == "text_delta" and chunk.text:
                full_text += chunk.text
                yield StreamEvent(type="text_delta", text=chunk.text)

            elif chunk.type == "tool_call_start" and chunk.tool_call:
                yield StreamEvent(
                    type="tool_call_start",
                    tool_name=chunk.tool_call.name,
                    tool_args=chunk.tool_call.arguments,
                )

            elif chunk.type == "tool_call_end" and chunk.tool_call:
                tool_calls_completed.append(chunk.tool_call)

            elif chunk.type == "usage" and chunk.usage:
                state.record_llm_call(chunk.usage.input_tokens, chunk.usage.output_tokens)

            elif chunk.type == "done":
                state._last_finish_reason = chunk.finish_reason or "stop"

        # Build and add assistant message
        from curio_agent_sdk.models.llm import ToolCall as TC
        assistant_msg = Message.assistant(
            content=full_text,
            tool_calls=tool_calls_completed if tool_calls_completed else None,
        )
        state.add_message(assistant_msg)

        # Execute tool calls if any
        if tool_calls_completed and self.tool_executor:
            state.record_tool_calls(len(tool_calls_completed))
            for tc in tool_calls_completed:
                result = await self.tool_executor.execute(tc)
                state.add_message(result.to_message())
                yield StreamEvent(
                    type="tool_call_end",
                    tool_name=tc.name,
                    tool_result=result.content,
                )

        state.iteration += 1
        yield StreamEvent(type="iteration_end", iteration=state.iteration)
