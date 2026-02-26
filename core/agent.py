"""
The main Agent class - a thin shell over Runtime.

Agent handles identity, configuration, and user-facing API.
Runtime handles all execution orchestration.

Agent supports two construction patterns:
1. Direct constructor:
    agent = Agent(model="openai:gpt-4o", tools=[search], system_prompt="...")

2. Fluent builder:
    agent = Agent.builder() \\
        .model("openai:gpt-4o") \\
        .tools([search]) \\
        .system_prompt("...") \\
        .build()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from curio_agent_sdk.core.builder import AgentBuilder
from curio_agent_sdk.core.context import ContextManager
from curio_agent_sdk.core.hooks import HookRegistry
from curio_agent_sdk.core.skills import Skill, SkillRegistry
from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.loops.tool_calling import ToolCallingLoop
from curio_agent_sdk.core.runtime import Runtime
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.models.llm import Message
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.models.events import AgentEvent, EventType, StreamEvent

if TYPE_CHECKING:
    from curio_agent_sdk.memory.manager import MemoryManager
    from curio_agent_sdk.core.state_store import StateStore

logger = logging.getLogger(__name__)


class Agent:
    """
    The primary agent class. A thin shell over Runtime.

    Agent handles:
    - Identity (agent_id, agent_name)
    - Component construction and wiring
    - User-facing API (run, arun, astream)
    - Builder pattern via Agent.builder()

    Runtime handles:
    - Loop execution
    - Memory injection/saving
    - State persistence
    - Event emission
    - Timeouts and cancellation

    Simple usage:
        agent = Agent(
            model="openai:gpt-4o",
            tools=[search, calculator],
            system_prompt="You are a helpful assistant.",
        )
        result = agent.run("What is the weather in SF?")

    Builder usage:
        agent = Agent.builder() \\
            .model("openai:gpt-4o") \\
            .tools([search, calculator]) \\
            .system_prompt("You are a helpful assistant.") \\
            .middleware([LoggingMiddleware(), CostTracker(budget=1.0)]) \\
            .memory_manager(MemoryManager(memory=ConversationMemory())) \\
            .build()
        result = await agent.arun("What is the weather in SF?")

    Advanced usage (direct Runtime access):
        # Access the runtime for custom orchestration
        state = agent.runtime.create_state("custom input")
        state.messages.insert(1, Message.system("Custom context"))
        result = await agent.runtime.run_with_state(state, agent_id=agent.agent_id)
    """

    def __init__(
        self,
        # Core
        system_prompt: str = "You are a helpful assistant.",
        tools: list[Tool | Callable] | None = None,
        loop: AgentLoop | None = None,
        llm: LLMClient | None = None,

        # Shorthand for simple setup (creates LLMClient automatically)
        model: str | None = None,
        tier: str = "tier2",

        # Identity
        agent_id: str | None = None,
        agent_name: str = "Agent",

        # Limits
        max_iterations: int = 25,
        timeout: float | None = None,
        iteration_timeout: float | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,

        # Context management
        context_manager: ContextManager | None = None,

        # Middleware
        middleware: list | None = None,

        # Human-in-the-loop
        human_input: Any | None = None,

        # Permissions / sandbox (optional)
        permission_policy: Any | None = None,

        # Memory
        memory_manager: MemoryManager | None = None,

        # State persistence
        state_store: StateStore | None = None,
        checkpoint_interval: int = 1,

        # Rules / instructions (merged into system_prompt)
        instruction_loader: Any | None = None,
        instructions: str | None = None,
        instructions_file: str | None = None,

        # Hooks & callbacks
        hook_registry: HookRegistry | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,

        # Skills (bundled tools + prompts + hooks; optional)
        skill_registry: SkillRegistry | None = None,
        skills: list[Skill] | None = None,

        # Subagent / multi-agent (optional)
        subagent_configs: dict[str, Any] | None = None,

        # Plan mode & todos (optional)
        plan_mode: Any | None = None,
        todo_manager: Any | None = None,
        read_only_tool_names: list[str] | None = None,

        # Session / conversation management (optional)
        session_manager: Any | None = None,

        # MCP (Model Context Protocol) integration (optional)
        mcp_server_urls: list[str] | None = None,
        mcp_server_configs: list[dict | Any] | None = None,
        mcp_resource_uris: list[str] | None = None,

        # Connector framework (optional)
        connectors: list[Any] | None = None,
    ):
        # ── Resolve instructions (direct construction) ─────────────────
        base_prompt = system_prompt
        if instruction_loader is not None:
            from curio_agent_sdk.core.instructions import InstructionLoader
            if isinstance(instruction_loader, InstructionLoader):
                loaded = instruction_loader.load()
                if loaded:
                    base_prompt = f"{base_prompt}\n\n---\n\n{loaded}"
        if instructions_file is not None:
            from curio_agent_sdk.core.instructions import load_instructions_from_file
            loaded = load_instructions_from_file(instructions_file)
            if loaded:
                base_prompt = f"{base_prompt}\n\n---\n\n{loaded}"
        if instructions is not None and instructions.strip():
            base_prompt = f"{base_prompt}\n\n---\n\n{instructions.strip()}"

        # ── Identity ────────────────────────────────────────────────
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name
        self.system_prompt = base_prompt

        # ── Hooks (lifecycle system; on_event is legacy, wired via adapter in Runtime) ──
        self.hook_registry = hook_registry if hook_registry is not None else HookRegistry()

        # ── Skills (build registry from list if provided) ───────────
        self.skill_registry = skill_registry
        if skills:
            if self.skill_registry is None:
                self.skill_registry = SkillRegistry()
            for s in skills:
                self.skill_registry.register(s)

        # ── Tools ───────────────────────────────────────────────────
        self.registry = ToolRegistry()
        for t in (tools or []):
            self.registry.register(t)
        if self.skill_registry:
            for skill in self.skill_registry.list():
                for t in skill.tools:
                    self.registry.register(t)
                for event, handler, priority in skill.hooks:
                    self.hook_registry.on(event, handler, priority=priority)
        # ── Subagent orchestrator (optional) ─────────────────────────
        self.orchestrator = None
        if subagent_configs:
            from curio_agent_sdk.core.subagent import AgentOrchestrator, SubagentConfig
            self.orchestrator = AgentOrchestrator(self)
            for name, cfg in subagent_configs.items():
                if not isinstance(cfg, SubagentConfig):
                    cfg = SubagentConfig(name=name, system_prompt=cfg.get("system_prompt", ""), tools=cfg.get("tools", []), model=cfg.get("model"), inherit_memory=cfg.get("inherit_memory", False), inherit_tools=cfg.get("inherit_tools", False), max_iterations=cfg.get("max_iterations", 10), timeout=cfg.get("timeout"))
                self.orchestrator.register(name, cfg)
            from curio_agent_sdk.core.tools.tool import tool
            orchestrator = self.orchestrator

            @tool
            async def spawn_subagent(task: str, agent_type: str = "general") -> str:
                """Spawn a subagent to handle a complex subtask. agent_type must be a registered subagent name."""
                result = await orchestrator.spawn(agent_type, task)
                return result.output or result.error or ""

            self.registry.register(spawn_subagent)

        # ── Plan mode & todos ─────────────────────────────────────────
        self.plan_mode = plan_mode
        self.todo_manager = todo_manager
        if plan_mode is not None or todo_manager is not None or read_only_tool_names is not None:
            from curio_agent_sdk.core.plan_mode import (
                PlanMode,
                TodoManager,
                get_plan_mode_tools,
            )
            pm = plan_mode
            tm = todo_manager
            if pm is None:
                pm = PlanMode(
                    read_only_tool_names=read_only_tool_names or [],
                    tool_registry=self.registry,
                )
            else:
                if getattr(pm, "tool_registry", None) is None:
                    pm.tool_registry = self.registry
                if read_only_tool_names is not None:
                    pm.read_only_tool_names = read_only_tool_names
            if tm is None:
                tm = TodoManager()
            self.plan_mode = pm
            self.todo_manager = tm
            for t in get_plan_mode_tools(pm, tm):
                self.registry.register(t)

        self.executor = ToolExecutor(
            self.registry,
            human_input=human_input,
            hook_registry=self.hook_registry,
            permission_policy=permission_policy,
        )

        # ── LLM Client ──────────────────────────────────────────────
        if llm:
            self.llm = llm
        elif model:
            provider, _, model_name = model.partition(":")
            from curio_agent_sdk.llm.router import TieredRouter
            router = TieredRouter()
            if provider and model_name:
                from curio_agent_sdk.llm.router import ModelPriority, TierConfig
                router.tiers[tier] = TierConfig(
                    name=tier,
                    model_priority=[ModelPriority(provider=provider, model=model_name)],
                )
            self.llm = LLMClient(router=router)
        else:
            self.llm = LLMClient()

        # ── Middleware wrapping ──────────────────────────────────────
        self.middleware = middleware or []
        if self.middleware:
            from curio_agent_sdk.middleware.base import MiddlewarePipeline
            self._middleware_pipeline = MiddlewarePipeline(
                self.middleware,
                hook_registry=self.hook_registry,
            )
            self.llm = self._middleware_pipeline.wrap_llm_client(self.llm)
        else:
            self._middleware_pipeline = None

        # ── Loop ────────────────────────────────────────────────────
        if loop:
            self.loop = loop
        else:
            self.loop = ToolCallingLoop(tier=tier, temperature=temperature, max_tokens=max_tokens)

        # Wire LLM and executor into the loop
        if hasattr(self.loop, 'llm') and self.loop.llm is None:
            self.loop.llm = self.llm
        if hasattr(self.loop, 'tool_executor') and self.loop.tool_executor is None:
            self.loop.tool_executor = self.executor

        # ── Store references for direct access ──────────────────────
        self.memory_manager_instance = memory_manager
        self.human_input = human_input
        self.context_manager = context_manager
        self.state_store = state_store
        self.checkpoint_interval = checkpoint_interval
        self.on_event = on_event
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.iteration_timeout = iteration_timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

        # ── MCP bridge (optional): connects at startup, registers tools ──
        mcp_bridge = None
        mcp_specs = list(mcp_server_urls or []) + list(mcp_server_configs or [])
        if mcp_specs:
            from curio_agent_sdk.mcp.bridge import MCPBridge
            mcp_bridge = MCPBridge(
                server_specs=mcp_specs,
                tool_registry=self.registry,
                resource_uris=mcp_resource_uris,
            )

        # ── Connector bridge (optional): connects at startup, registers tools ──
        connector_bridge = None
        if connectors:
            from curio_agent_sdk.connectors.bridge import ConnectorBridge
            connector_bridge = ConnectorBridge(
                connectors=list(connectors),
                tool_registry=self.registry,
            )

        # ── Build Runtime ───────────────────────────────────────────
        self.runtime = Runtime(
            loop=self.loop,
            llm=self.llm,
            tool_registry=self.registry,
            tool_executor=self.executor,
            system_prompt=self.system_prompt,
            max_iterations=max_iterations,
            timeout=timeout,
            iteration_timeout=iteration_timeout,
            context_manager=context_manager,
            memory_manager=memory_manager,
            state_store=state_store,
            checkpoint_interval=checkpoint_interval,
            hook_registry=self.hook_registry,
            on_event=on_event,
            skill_registry=self.skill_registry,
            plan_mode=getattr(self, "plan_mode", None),
            todo_manager=getattr(self, "todo_manager", None),
            session_manager=session_manager,
            mcp_bridge=mcp_bridge,
            connector_bridge=connector_bridge,
        )
        self.session_manager = session_manager

        # Expose the resolved memory_manager from runtime
        if self.runtime.memory_manager is not None:
            self.memory_manager_instance = self.runtime.memory_manager

    # ── Builder pattern ─────────────────────────────────────────────

    @classmethod
    def builder(cls) -> AgentBuilder:
        """
        Create a fluent builder for constructing an Agent.

        Example:
            agent = Agent.builder() \\
                .model("openai:gpt-4o") \\
                .tools([search]) \\
                .system_prompt("You are helpful.") \\
                .build()

        Returns:
            AgentBuilder instance for fluent configuration.
        """
        return AgentBuilder()

    # ── Execution API (delegates to Runtime) ────────────────────────

    async def arun(
        self,
        input: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
        resume_from: str | None = None,
        active_skills: list[str] | None = None,
        response_format: type | dict[str, Any] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentRunResult:
        """
        Run the agent asynchronously.

        Args:
            input: The user's input/objective.
            context: Optional additional context dict.
            max_iterations: Override max iterations for this run.
            timeout: Override timeout for this run (seconds).
            resume_from: Optional run_id to resume from saved state.
            response_format: Optional Pydantic model or list[Model] for structured
                output; result.parsed_output will hold the validated instance(s).
            session_id: Optional session ID for multi-turn conversation; loads history
                and persists new messages (requires session_manager on the agent).
            run_id: Optional explicit run ID (e.g. for TaskManager to use task_id for checkpoint correlation).

        Returns:
            AgentRunResult with status, output, and metrics (and parsed_output if response_format used).
        """
        return await self.runtime.run(
            input,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            context=context,
            max_iterations=max_iterations,
            timeout=timeout,
            resume_from=resume_from,
            active_skills=active_skills,
            response_format=response_format,
            session_id=session_id,
            session_manager=self.session_manager,
            run_id=run_id,
        )

    async def invoke_skill(
        self,
        name: str,
        input: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
    ) -> AgentRunResult:
        """
        Run the agent with a single skill active (tools + prompt for that skill).

        Equivalent to arun(input, active_skills=[name], ...). Use when you want
        to scope the run to one skill (e.g. "commit", "review-pr").

        Args:
            name: Skill name (must be registered on this agent).
            input: User input / task for this skill.
            context: Optional additional context dict.
            max_iterations: Override max iterations.
            timeout: Override timeout (seconds).

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        if self.skill_registry is None or self.skill_registry.get(name) is None:
            raise ValueError(f"Unknown skill: {name}. Registered skills: {self.skill_registry.list_names() if self.skill_registry else []}")
        return await self.arun(
            input,
            context=context,
            max_iterations=max_iterations,
            timeout=timeout,
            active_skills=[name],
        )

    # ── Subagent / multi-agent ───────────────────────────────────────

    async def spawn_subagent(
        self,
        config: str | Any,
        task: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
    ) -> AgentRunResult:
        """
        Spawn a subagent with the given config (or registered name), run it on task, return result.

        Requires subagent_configs to have been provided at build time (or orchestrator set).
        """
        if self.orchestrator is None:
            raise RuntimeError("No orchestrator: pass subagent_configs when building the agent.")
        return await self.orchestrator.spawn(config, task, context=context, max_iterations=max_iterations, timeout=timeout)

    async def spawn_subagent_background(
        self,
        config: str | Any,
        task: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Spawn a subagent in the background. Returns task_id; use get_subagent_result(task_id) to get the result."""
        if self.orchestrator is None:
            raise RuntimeError("No orchestrator: pass subagent_configs when building the agent.")
        return await self.orchestrator.spawn_background(config, task, context=context, max_iterations=max_iterations, timeout=timeout)

    async def get_subagent_result(self, task_id: str) -> AgentRunResult | None:
        """Return the result of a background subagent run if completed, else None."""
        if self.orchestrator is None:
            return None
        return await self.orchestrator.get_result(task_id)

    async def handoff(
        self,
        target: "Agent",
        context: str,
        parent_messages: list | None = None,
        agent_id: str = "",
        run_id: str | None = None,
    ) -> AgentRunResult:
        """Hand off the conversation to another agent. Optionally pass parent message history."""
        if self.orchestrator is not None:
            return await self.orchestrator.handoff(target, context, parent_messages=parent_messages, agent_id=agent_id, run_id=run_id)
        if parent_messages is None or len(parent_messages) == 0:
            return await target.arun(context)
        from curio_agent_sdk.core.state import AgentState
        from curio_agent_sdk.models.llm import Message
        messages = list(parent_messages) + [Message.user(context)]
        state = AgentState(
            messages=messages,
            tools=target.registry.tools,
            tool_schemas=target.registry.get_llm_schemas(),
            max_iterations=getattr(target, "max_iterations", 25),
        )
        return await target.runtime.run_with_state(state, agent_id=agent_id or target.agent_id, run_id=run_id)

    async def astream(
        self,
        input: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream agent execution events.

        Yields StreamEvent objects for real-time observation of agent activity.

        Args:
            input: The user's input/objective.
            context: Optional additional context dict.

        Yields:
            StreamEvent objects.
        """
        async for event in self.runtime.stream(
            input,
            agent_id=self.agent_id,
            context=context,
        ):
            yield event

    def run(
        self,
        input: str,
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
        resume_from: str | None = None,
        active_skills: list[str] | None = None,
        response_format: type | dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> AgentRunResult:
        """
        Run the agent synchronously. Convenience wrapper around arun().

        Args:
            input: The user's input/objective.
            context: Optional additional context dict.
            max_iterations: Override max iterations.
            timeout: Override timeout (seconds).
            resume_from: Optional run_id to resume from saved state.
            response_format: Optional Pydantic model or list[Model] for structured output.
            session_id: Optional session ID for multi-turn conversation.

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.arun(
                        input,
                        context=context,
                        max_iterations=max_iterations,
                        timeout=timeout,
                        resume_from=resume_from,
                        active_skills=active_skills,
                        response_format=response_format,
                        session_id=session_id,
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    input,
                    context=context,
                    max_iterations=max_iterations,
                    timeout=timeout,
                    resume_from=resume_from,
                    active_skills=active_skills,
                    response_format=response_format,
                    session_id=session_id,
                )
            )

    # ── Component lifecycle ────────────────────────────────────────

    async def start(self) -> None:
        """
        Start all components (memory, state store, etc.).

        Idempotent; safe to call multiple times. Called automatically
        before the first run()/arun()/astream() if not called explicitly.
        """
        await self.runtime.startup_components()
        self.runtime._components_started = True

    async def close(self) -> None:
        """
        Shut down all components (memory, state store, etc.).

        Call when done with the agent to release resources. Safe to
        call multiple times. Use `async with agent:` to close automatically.
        """
        await self.runtime.shutdown_components()

    async def __aenter__(self) -> Agent:
        """Start components when entering async with block."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shut down components when exiting async with block."""
        await self.close()

    # ── Plan mode & todos ───────────────────────────────────────────

    def is_in_plan_mode(self) -> bool:
        """Return True if the agent is in plan mode (read-only exploration). Uses state from current/last run."""
        if self.plan_mode is None:
            return False
        state = getattr(self.runtime, "_last_state", None)
        return self.plan_mode.is_in_plan_mode(state)

    def is_awaiting_plan_approval(self) -> bool:
        """Return True if the agent has submitted a plan and is waiting for approval."""
        if self.plan_mode is None:
            return False
        state = getattr(self.runtime, "_last_state", None)
        return self.plan_mode.is_awaiting_approval(state)

    def get_plan(self) -> Any | None:
        """Return the current plan if any (from PlanState). None if no plan mode or no plan."""
        if self.plan_mode is None:
            return None
        state = getattr(self.runtime, "_last_state", None)
        return self.plan_mode.get_plan(state)

    # ── Rules / instructions ────────────────────────────────────────

    def add_instructions(self, text: str) -> None:
        """
        Append instructions to be injected into the system prompt on the next run.
        Use for dynamic instruction injection (e.g. rules added mid-session).
        """
        self.runtime.add_instructions(text)

    def clear_extra_instructions(self) -> None:
        """Clear any dynamically added instructions."""
        self.runtime.clear_extra_instructions()

    # ── Convenience properties ──────────────────────────────────────

    @property
    def tools(self) -> list[Tool]:
        """Get the agent's registered tools."""
        return self.registry.tools

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id!r}, name={self.agent_name!r}, "
            f"tools={len(self.registry.tools)}, loop={self.loop.__class__.__name__})"
        )
