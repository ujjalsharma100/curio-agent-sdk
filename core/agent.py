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

        # ── Tools ───────────────────────────────────────────────────
        self.registry = ToolRegistry()
        for t in (tools or []):
            self.registry.register(t)
        self.executor = ToolExecutor(
            self.registry,
            human_input=human_input,
            hook_registry=self.hook_registry,
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
        )

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
    ) -> AgentRunResult:
        """
        Run the agent asynchronously.

        Args:
            input: The user's input/objective.
            context: Optional additional context dict.
            max_iterations: Override max iterations for this run.
            timeout: Override timeout for this run (seconds).
            resume_from: Optional run_id to resume from saved state.

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        return await self.runtime.run(
            input,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            context=context,
            max_iterations=max_iterations,
            timeout=timeout,
            resume_from=resume_from,
        )

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
    ) -> AgentRunResult:
        """
        Run the agent synchronously. Convenience wrapper around arun().

        Args:
            input: The user's input/objective.
            context: Optional additional context dict.
            max_iterations: Override max iterations.
            timeout: Override timeout (seconds).
            resume_from: Optional run_id to resume from saved state.

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.arun(input, context, max_iterations, timeout, resume_from),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.arun(input, context, max_iterations, timeout, resume_from))

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
