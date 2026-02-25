"""
Plan-Critique-Synthesize loop.

The original Curio Agent pattern, now as a composable loop:
1. Plan: LLM generates a list of actions
2. Execute: Run each planned tool call
3. Critique: LLM evaluates progress and decides to continue or stop
4. Repeat until done
5. Synthesize: LLM produces final summary

This loop uses native tool calling for the execution phase but
uses structured prompting for plan/critique/synthesis phases.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.models.llm import LLMRequest, Message, ToolCall
from curio_agent_sdk.models.events import StreamEvent

logger = logging.getLogger(__name__)


PLAN_INSTRUCTIONS = """Based on the objective and available tools, create a plan of actions.
Return a JSON object with this exact format:
{
    "plan": [
        {"action": "<tool_name>", "args": {"param1": "value1"}}
    ],
    "notes": "Brief notes on the plan"
}
Set plan to empty list if no action is needed."""

CRITIQUE_INSTRUCTIONS = """Evaluate what has been done so far.
Return a JSON object with this exact format:
{
    "status": "done" or "continue",
    "summary": "Brief evaluation",
    "recommendations": "What to do next if continuing"
}"""

SYNTHESIS_INSTRUCTIONS = """Synthesize a clear summary of what was accomplished during this run.
Return only the summary text, no JSON."""


class PlanCritiqueSynthesizeLoop(AgentLoop):
    """
    Plan-Critique-Synthesize agentic loop.

    Each iteration:
    1. Ask LLM to plan actions (structured JSON output)
    2. Execute each planned action via tool registry
    3. Ask LLM to critique progress
    4. Continue if critique says "continue", stop if "done"

    After loop ends:
    5. Ask LLM to synthesize final output

    This uses 3 LLM calls per iteration (plan + critique + synthesis at end),
    which is more expensive but provides structured reasoning.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tool_executor: ToolExecutor | None = None,
        plan_tier: str = "tier3",
        critique_tier: str = "tier3",
        synthesis_tier: str = "tier1",
        action_tier: str = "tier2",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        run_id: str | None = None,
        agent_id: str | None = None,
    ):
        self.llm = llm
        self.tool_executor = tool_executor
        self.plan_tier = plan_tier
        self.critique_tier = critique_tier
        self.synthesis_tier = synthesis_tier
        self.action_tier = action_tier
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.run_id = run_id
        self.agent_id = agent_id
        self._critique_status = "continue"

    def _fit_messages(self, messages: list[Message]) -> list[Message]:
        """Apply context window management if a context_manager is set."""
        if self.context_manager is not None:
            return self.context_manager.fit_messages(messages)
        return messages

    async def step(self, state: AgentState) -> AgentState:
        """One iteration: plan -> execute -> critique."""
        if not self.llm:
            raise RuntimeError("LLMClient not set on PlanCritiqueSynthesizeLoop")

        # === PLAN ===
        plan_messages = list(state.messages) + [
            Message.user(PLAN_INSTRUCTIONS)
        ]

        # Add tool descriptions to help planning
        tool_desc = self._format_tools(state)
        if tool_desc:
            plan_messages.insert(-1, Message.user(f"Available tools:\n{tool_desc}"))

        # Fit messages within context window
        plan_messages = self._fit_messages(plan_messages)

        plan_request = LLMRequest(
            messages=plan_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tier=self.plan_tier,
            response_format={"type": "json_object"},
        )

        plan_response = await self.llm.call(plan_request, run_id=self.run_id, agent_id=self.agent_id)
        state.record_llm_call(plan_response.usage.input_tokens, plan_response.usage.output_tokens)

        # Parse plan
        planned_actions = self._parse_plan(plan_response.content)

        if not planned_actions:
            logger.info("Empty plan returned, marking as done")
            self._critique_status = "done"
            state.add_message(Message.assistant("No further actions needed."))
            state.iteration += 1
            return state

        # Record plan in messages
        state.add_message(Message.assistant(
            f"Plan: {json.dumps([{'action': a['action'], 'args': a['args']} for a in planned_actions])}"
        ))

        # === EXECUTE ===
        if self.tool_executor:
            for action in planned_actions:
                tool_call = ToolCall(
                    id=f"plan_{state.iteration}_{action['action']}",
                    name=action["action"],
                    arguments=action["args"],
                )
                result = await self.tool_executor.execute(tool_call)
                state.record_tool_calls(1)

                # Add execution result to messages
                state.add_message(Message.user(
                    f"Executed {action['action']}: {result.content}"
                ))

        # === CRITIQUE ===
        critique_messages = list(state.messages) + [
            Message.user(CRITIQUE_INSTRUCTIONS)
        ]

        # Fit messages within context window
        critique_messages = self._fit_messages(critique_messages)

        critique_request = LLMRequest(
            messages=critique_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tier=self.critique_tier,
            response_format={"type": "json_object"},
        )

        critique_response = await self.llm.call(critique_request, run_id=self.run_id, agent_id=self.agent_id)
        state.record_llm_call(critique_response.usage.input_tokens, critique_response.usage.output_tokens)

        # Parse critique
        self._critique_status = self._parse_critique(critique_response.content)

        state.add_message(Message.assistant(f"Critique: {critique_response.content}"))
        state.iteration += 1

        return state

    def should_continue(self, state: AgentState) -> bool:
        if state.is_cancelled or state.is_done:
            return False
        if state.iteration >= state.max_iterations:
            return False
        return self._critique_status == "continue"

    def get_output(self, state: AgentState) -> str:
        """For plan-critique, we look for synthesis output or last content."""
        # Look for synthesis message (added by the agent after loop ends)
        for msg in reversed(state.messages):
            if msg.role == "assistant" and msg.content:
                text = msg.text
                if text and not text.startswith("Plan:") and not text.startswith("Critique:"):
                    return text
        return ""

    async def synthesize(self, state: AgentState) -> str:
        """
        Run the synthesis phase after the loop completes.

        Called by the Agent class after should_continue returns False.
        """
        if not self.llm:
            return self.get_output(state)

        synthesis_messages = list(state.messages) + [
            Message.user(SYNTHESIS_INSTRUCTIONS)
        ]

        # Fit messages within context window
        synthesis_messages = self._fit_messages(synthesis_messages)

        synthesis_request = LLMRequest(
            messages=synthesis_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tier=self.synthesis_tier,
        )

        response = await self.llm.call(synthesis_request, run_id=self.run_id, agent_id=self.agent_id)
        state.record_llm_call(response.usage.input_tokens, response.usage.output_tokens)

        # Add synthesis to messages
        state.add_message(response.message)

        return response.content

    def _format_tools(self, state: AgentState) -> str:
        """Format tool descriptions for the planning prompt."""
        if not state.tools:
            return ""
        lines = []
        for tool in state.tools:
            params = ", ".join(
                f"{p.name}: {p.type}" + (f" (default: {p.default})" if p.default is not None else "")
                for p in tool.schema.parameters
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def _parse_plan(self, content: str) -> list[dict]:
        """Parse planned actions from LLM response."""
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            plan = data.get("plan", [])
            return [{"action": a.get("action", ""), "args": a.get("args", {})} for a in plan if a.get("action")]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse plan: {e}")
            return []

    def _parse_critique(self, content: str) -> str:
        """Parse critique status from LLM response."""
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            return data.get("status", "done").lower()
        except (json.JSONDecodeError, KeyError, TypeError):
            return "done"
