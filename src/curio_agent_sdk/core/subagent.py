"""
Subagent / multi-agent composition.

SubagentConfig defines a named subagent (prompt, tools, model, inheritance).
AgentOrchestrator manages spawning subagents from a parent agent, background
tasks, and handoffs to other agents.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.models.llm import Message

if TYPE_CHECKING:
    from curio_agent_sdk.core.agent import Agent
    from curio_agent_sdk.core.tools.tool import Tool

logger = logging.getLogger(__name__)


@dataclass
class SubagentConfig:
    """
    Configuration for spawning a subagent.

    When model is None, the subagent uses the parent agent's LLM.
    When inherit_tools is True, the subagent gets the parent's tools in addition to its own.
    When inherit_memory is True, the subagent shares the parent's memory manager.
    When inherit_hooks is True (default), the subagent shares the parent's HookRegistry,
    so that hook consumers (tracing, logging, persistence) automatically cover subagent
    activity and OTel spans share the same trace_id.
    """

    name: str
    system_prompt: str
    tools: list[Any] = field(default_factory=list)  # list[Tool] but avoid circular import
    model: str | None = None
    inherit_memory: bool = False
    inherit_tools: bool = False
    inherit_hooks: bool = True
    max_iterations: int = 10
    timeout: float | None = None


class AgentOrchestrator:
    """
    Manages subagent spawning and communication from a parent agent.

    - spawn(config, task): run a subagent with the given config and task string; return result.
    - spawn_background(config, task): start subagent in background; return task_id.
    - get_result(task_id): return result of a background run if done, else None.
    - handoff(target_agent, context, parent_messages): pass conversation to another agent.
    """

    def __init__(self, parent: "Agent") -> None:
        self._parent = parent
        self._configs: dict[str, SubagentConfig] = {}
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, AgentRunResult] = {}

    def register(self, name: str, config: SubagentConfig) -> None:
        """Register a subagent config by name."""
        if config.name != name:
            config = SubagentConfig(
                name=name,
                system_prompt=config.system_prompt,
                tools=config.tools,
                model=config.model,
                inherit_memory=config.inherit_memory,
                inherit_tools=config.inherit_tools,
                max_iterations=config.max_iterations,
                timeout=config.timeout,
            )
        self._configs[name] = config

    def get_config(self, name: str) -> SubagentConfig | None:
        """Return the config for a registered subagent name."""
        return self._configs.get(name)

    def list_names(self) -> list[str]:
        """Return names of all registered subagent configs."""
        return list(self._configs.keys())

    def _build_subagent(self, config: SubagentConfig) -> "Agent":
        """Build an Agent instance from a SubagentConfig using parent's resources."""
        from curio_agent_sdk.core.agent import Agent

        tools = list(config.tools)
        if config.inherit_tools and self._parent.registry:
            for t in self._parent.registry.tools:
                if t not in tools:
                    tools.append(t)

        memory_manager = None
        if config.inherit_memory and getattr(
            self._parent, "memory_manager_instance", None
        ):
            memory_manager = self._parent.memory_manager_instance

        # Inherit hook registry for consumers (tracing, logging, etc.)
        hook_registry = None
        if config.inherit_hooks and getattr(self._parent, "hook_registry", None):
            hook_registry = self._parent.hook_registry

        kwargs: dict[str, Any] = dict(
            system_prompt=config.system_prompt,
            tools=tools,
            memory_manager=memory_manager,
            max_iterations=config.max_iterations,
            timeout=config.timeout,
        )
        if hook_registry is not None:
            kwargs["hook_registry"] = hook_registry

        if config.model:
            return Agent(model=config.model, **kwargs)
        return Agent(llm=self._parent.llm, **kwargs)

    async def spawn(
        self,
        config: SubagentConfig | str,
        task: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
    ) -> AgentRunResult:
        """
        Spawn a subagent, run it with the given task, and return the result.

        Args:
            config: SubagentConfig or registered subagent name.
            task: The task string (user input) for the subagent.
            context: Optional additional context dict.
            max_iterations: Override config's max_iterations for this run.
            timeout: Override config's timeout for this run.

        Returns:
            AgentRunResult from the subagent run.
        """
        if isinstance(config, str):
            cfg = self._configs.get(config)
            if cfg is None:
                raise ValueError(
                    f"Unknown subagent: {config}. Registered: {list(self._configs.keys())}"
                )
            config = cfg

        # Capture current OTel context so subagent spans share the same trace_id
        otel_token = None
        try:
            from opentelemetry import context as otel_context
            parent_ctx = otel_context.get_current()
            otel_token = otel_context.attach(parent_ctx)
        except ImportError:
            pass

        subagent = self._build_subagent(config)
        try:
            if getattr(self._parent, "runtime", None) and getattr(
                self._parent.runtime, "_components_started", False
            ):
                await subagent.start()
            return await subagent.arun(
                task,
                context=context,
                max_iterations=max_iterations or config.max_iterations,
                timeout=timeout or config.timeout,
            )
        finally:
            await subagent.close()
            if otel_token is not None:
                try:
                    from opentelemetry import context as otel_context
                    otel_context.detach(otel_token)
                except ImportError:
                    pass

    async def spawn_background(
        self,
        config: SubagentConfig | str,
        task: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """
        Spawn a subagent in the background. Returns a task_id to poll with get_result().

        Args:
            config: SubagentConfig or registered subagent name.
            task: The task string for the subagent.
            context: Optional context dict.
            max_iterations: Override for this run.
            timeout: Override for this run.

        Returns:
            task_id: Use get_result(task_id) to retrieve the result.
        """
        task_id = str(uuid.uuid4())

        async def _run() -> None:
            try:
                result = await self.spawn(
                    config, task,
                    context=context,
                    max_iterations=max_iterations,
                    timeout=timeout,
                )
                self._results[task_id] = result
            except Exception as e:
                logger.exception("Background subagent run failed: %s", e)
                from curio_agent_sdk.models.agent import AgentRunResult
                self._results[task_id] = AgentRunResult(
                    status="error",
                    output="",
                    error=str(e),
                    run_id=task_id,
                )
            finally:
                self._background_tasks.pop(task_id, None)

        self._background_tasks[task_id] = asyncio.create_task(_run())
        return task_id

    async def get_result(self, task_id: str) -> AgentRunResult | None:
        """
        Get the result of a background subagent run if it has completed.

        Returns:
            AgentRunResult if the run is done, None otherwise.
        """
        if task_id in self._results:
            return self._results.pop(task_id, None)
        t = self._background_tasks.get(task_id)
        if t is None:
            return None
        if t.done():
            self._background_tasks.pop(task_id, None)
            return self._results.pop(task_id, None)
        return None

    async def handoff(
        self,
        target: "Agent",
        context: str,
        parent_messages: list[Message] | None = None,
        agent_id: str = "",
        run_id: str | None = None,
    ) -> AgentRunResult:
        """
        Hand off the conversation to another agent.

        If parent_messages is provided, the target agent runs with that message
        history plus a final user message containing context. Otherwise the
        target runs with context as the only user input.

        Args:
            target: The agent to hand off to.
            context: Handoff context (e.g. "Review this code and suggest improvements").
            parent_messages: Optional conversation history from the parent run.
            agent_id: Optional agent ID for the handoff run.
            run_id: Optional run ID.

        Returns:
            AgentRunResult from the target agent.
        """
        if parent_messages is None or len(parent_messages) == 0:
            return await target.arun(context)
        # Build state with parent messages + handoff context
        from curio_agent_sdk.core.state import AgentState

        messages = list(parent_messages) + [Message.user(context)]
        state = AgentState(
            messages=messages,
            tools=target.registry.tools,
            tool_schemas=target.registry.get_llm_schemas(),
            max_iterations=getattr(target, "max_iterations", 25),
        )
        return await target.runtime.run_with_state(
            state,
            agent_id=agent_id or target.agent_id,
            run_id=run_id,
        )
