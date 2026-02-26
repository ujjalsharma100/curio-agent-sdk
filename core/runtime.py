"""
Runtime - the orchestration engine extracted from Agent.

Handles all execution logic: loop driving, memory injection/saving,
state persistence, event emission, timeouts, and cancellation.

The Agent class is a thin shell that delegates to Runtime.
Runtime is independently usable for advanced use cases.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from curio_agent_sdk.core.component import Component
from curio_agent_sdk.core.context import ContextManager
from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.models.llm import Message
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.models.events import AgentEvent, EventType, StreamEvent

if TYPE_CHECKING:
    from curio_agent_sdk.memory.manager import MemoryManager
    from curio_agent_sdk.core.state_store import StateStore
    from curio_agent_sdk.llm.client import LLMClient

logger = logging.getLogger(__name__)


class Runtime:
    """
    The orchestration engine that drives agent execution.

    Runtime is responsible for:
    - Creating initial agent state from user input
    - Driving the agent loop (step -> should_continue -> repeat)
    - Memory management via MemoryManager (injection, saving, per-iteration)
    - Persisting state periodically via StateStore
    - Emitting lifecycle events for observability
    - Enforcing timeouts and cancellation

    This was extracted from the Agent class to separate concerns:
    - Agent = identity + configuration + user-facing API
    - Runtime = execution engine

    Runtime can be used independently for advanced scenarios like
    custom orchestrators, multi-agent systems, or testing.

    Memory handling:
    - Pass a MemoryManager for full control over injection/save/query strategies
    - Pass None to disable memory entirely

    Example:
        runtime = Runtime(
            loop=ToolCallingLoop(tier="tier2"),
            llm=my_llm_client,
            tool_registry=my_registry,
            tool_executor=my_executor,
            system_prompt="You are helpful.",
        )
        result = await runtime.run("Hello!", agent_id="my-agent")

    Example with custom memory strategies:
        from curio_agent_sdk.memory.manager import (
            MemoryManager, UserMessageInjection, SaveEverythingStrategy,
        )

        manager = MemoryManager(
            memory=VectorMemory(),
            injection_strategy=UserMessageInjection(max_tokens=4000),
            save_strategy=SaveEverythingStrategy(),
        )
        runtime = Runtime(
            ...,
            memory_manager=manager,
        )
    """

    def __init__(
        self,
        # Core (required for execution)
        loop: AgentLoop,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        system_prompt: str = "You are a helpful assistant.",

        # Limits
        max_iterations: int = 25,
        timeout: float | None = None,
        iteration_timeout: float | None = None,

        # Context management
        context_manager: ContextManager | None = None,

        # Memory
        memory_manager: MemoryManager | None = None,

        # State persistence
        state_store: StateStore | None = None,
        checkpoint_interval: int = 1,

        # Callbacks
        on_event: Callable[[AgentEvent], None] | None = None,
    ):
        self.loop = loop
        self.llm = llm
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.iteration_timeout = iteration_timeout
        self.context_manager = context_manager
        self.state_store = state_store
        self.checkpoint_interval = checkpoint_interval
        self.on_event = on_event

        # Memory manager (no auto-wrapping; pass MemoryManager explicitly)
        self.memory_manager = memory_manager

        # Wire context manager into the loop
        if self.context_manager is not None:
            self.loop.context_manager = self.context_manager

        # Component lifecycle: start once before first run, shutdown on close
        self._components_started = False

    # ── Component lifecycle ─────────────────────────────────────────

    def _gather_components(self) -> list[Component]:
        """Collect all dependencies that implement Component (unique by id)."""
        candidates = [
            self.memory_manager,
            self.state_store,
            self.llm,
            self.loop,
        ]
        seen: set[int] = set()
        out: list[Component] = []
        for obj in candidates:
            if obj is not None and isinstance(obj, Component):
                oid = id(obj)
                if oid not in seen:
                    seen.add(oid)
                    out.append(obj)
        return out

    async def startup_components(self) -> None:
        """Start all components that support lifecycle. Idempotent after first call."""
        for comp in self._gather_components():
            try:
                await comp.startup()
                logger.debug("Started component %s", type(comp).__name__)
            except Exception as e:
                logger.warning("Component %s startup failed: %s", type(comp).__name__, e)

    async def shutdown_components(self) -> None:
        """Shut down all components. Safe to call multiple times."""
        for comp in self._gather_components():
            try:
                await comp.shutdown()
                logger.debug("Shut down component %s", type(comp).__name__)
            except Exception as e:
                logger.warning("Component %s shutdown failed: %s", type(comp).__name__, e)
        self._components_started = False

    async def _ensure_components_started(self) -> None:
        """Ensure all components are started (called at start of run/stream)."""
        if self._components_started:
            return
        await self.startup_components()
        self._components_started = True

    async def health_check_components(self) -> dict[str, bool]:
        """Run health_check on all components. Returns name -> healthy."""
        result: dict[str, bool] = {}
        for comp in self._gather_components():
            name = type(comp).__name__
            try:
                result[name] = await comp.health_check()
            except Exception as e:
                logger.warning("Health check %s failed: %s", name, e)
                result[name] = False
        return result

    # ── State creation ──────────────────────────────────────────────

    def create_state(self, input_text: str, context: dict[str, Any] | None = None) -> AgentState:
        """
        Create initial agent state for a run.

        Public so that advanced users can create state, modify it,
        then pass to run_with_state().
        """
        messages = [Message.system(self.system_prompt)]

        if context:
            import json
            input_with_context = f"{input_text}\n\nAdditional context:\n{json.dumps(context, indent=2)}"
            messages.append(Message.user(input_with_context))
        else:
            messages.append(Message.user(input_text))

        return AgentState(
            messages=messages,
            tools=self.tool_registry.tools,
            tool_schemas=self.tool_registry.get_llm_schemas(),
            max_iterations=self.max_iterations,
        )

    # ── Memory (delegates to MemoryManager) ─────────────────────────

    async def inject_memory_context(self, state: AgentState, input_text: str) -> None:
        """
        Inject memory context into the agent state.

        Delegates to MemoryManager.inject() which uses the configured
        injection strategy and query strategy.
        """
        if self.memory_manager is None:
            return
        await self.memory_manager.inject(state, input_text)

    async def save_to_memory(self, state: AgentState, input_text: str, output: str) -> None:
        """
        Save the interaction to memory after a successful run.

        Delegates to MemoryManager.on_run_end() which uses the configured
        save strategy.
        """
        if self.memory_manager is None:
            return
        await self.memory_manager.on_run_end(input_text, output, state)

    async def _memory_on_run_start(self, input_text: str, state: AgentState) -> None:
        """Notify memory manager of run start."""
        if self.memory_manager is None:
            return
        await self.memory_manager.on_run_start(input_text, state)

    async def _memory_on_run_error(self, input_text: str, error: str, state: AgentState) -> None:
        """Notify memory manager of run error."""
        if self.memory_manager is None:
            return
        await self.memory_manager.on_run_error(input_text, error, state)

    async def _memory_on_iteration(self, state: AgentState, iteration: int) -> None:
        """Notify memory manager of iteration completion."""
        if self.memory_manager is None:
            return
        await self.memory_manager.on_iteration(state, iteration)

    # ── State persistence ───────────────────────────────────────────

    async def save_state(self, state: AgentState, run_id: str, agent_id: str) -> None:
        """Save agent state via the StateStore."""
        if self.state_store is None:
            return

        try:
            await self.state_store.save(state, run_id, agent_id)
            self._emit(EventType.CHECKPOINT_SAVED, run_id, agent_id, state.iteration)
        except Exception as e:
            logger.warning("Failed to save state: %s", e)

    async def restore_state(self, run_id: str, agent_id: str) -> AgentState | None:
        """Restore agent state from the StateStore."""
        if self.state_store is None:
            return None

        try:
            state = await self.state_store.load(run_id)
            if state is None:
                return None

            # Re-attach tools from registry
            state.tools = self.tool_registry.tools
            state.tool_schemas = self.tool_registry.get_llm_schemas()
            state.max_iterations = self.max_iterations

            self._emit(EventType.CHECKPOINT_RESTORED, run_id, agent_id, state.iteration)
            logger.info("Restored state for run %s at iteration %d", run_id, state.iteration)
            return state

        except Exception as e:
            logger.warning("Failed to restore state for run %s: %s", run_id, e)
            return None

    # ── Event emission ──────────────────────────────────────────────

    def _emit(self, event_type: EventType, run_id: str, agent_id: str, iteration: int = 0, **data):
        """Emit an agent event."""
        if self.on_event:
            try:
                event = AgentEvent(
                    type=event_type,
                    run_id=run_id,
                    agent_id=agent_id,
                    iteration=iteration,
                    data=data,
                )
                self.on_event(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

    # ── Core execution ──────────────────────────────────────────────

    async def run(
        self,
        input: str,
        *,
        agent_id: str = "",
        agent_name: str = "Agent",
        context: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        timeout: float | None = None,
        resume_from: str | None = None,
    ) -> AgentRunResult:
        """
        Run the agent loop to completion.

        This is the main execution method. It:
        1. Creates or restores state
        2. Injects memory context (via MemoryManager)
        3. Notifies memory of run start
        4. Drives the loop until done, timeout, cancel, or error
        5. Notifies memory per-iteration
        6. Extracts output
        7. Saves to memory (via MemoryManager)
        8. Returns the result

        Args:
            input: The user's input/objective.
            agent_id: ID of the agent (for events and state persistence).
            agent_name: Display name of the agent.
            context: Optional additional context dict.
            max_iterations: Override max iterations for this run.
            timeout: Override timeout for this run (seconds).
            resume_from: Optional run_id to resume from saved state.

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        run_id = resume_from or str(uuid.uuid4())
        effective_timeout = timeout or self.timeout

        await self._ensure_components_started()

        # Try to resume from saved state
        state = None
        if resume_from:
            state = await self.restore_state(resume_from, agent_id)

        # Create fresh state if not resuming
        if state is None:
            state = self.create_state(input, context)
            await self.inject_memory_context(state, input)

        if max_iterations:
            state.max_iterations = max_iterations

        # Set run_id/agent_id on loop if supported
        if hasattr(self.loop, 'run_id'):
            self.loop.run_id = run_id
        if hasattr(self.loop, 'agent_id'):
            self.loop.agent_id = agent_id

        self._emit(EventType.RUN_STARTED, run_id, agent_id, data={
            "input": input,
            "resumed": resume_from is not None,
        })

        # Notify memory of run start
        await self._memory_on_run_start(input, state)

        try:
            if effective_timeout:
                state = await asyncio.wait_for(
                    self._execute_loop(state, run_id, agent_id),
                    timeout=effective_timeout,
                )
            else:
                state = await self._execute_loop(state, run_id, agent_id)

        except asyncio.TimeoutError:
            self._emit(EventType.RUN_TIMEOUT, run_id, agent_id, state.iteration,
                       elapsed=state.elapsed_time)
            await self.save_state(state, run_id, agent_id)
            return AgentRunResult(
                status="timeout",
                output="",
                total_iterations=state.iteration,
                total_llm_calls=state.total_llm_calls,
                total_tool_calls=state.total_tool_calls,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                run_id=run_id,
                error=f"Agent timed out after {effective_timeout}s",
                messages=state.messages,
            )

        except asyncio.CancelledError:
            self._emit(EventType.RUN_CANCELLED, run_id, agent_id, state.iteration,
                       elapsed=state.elapsed_time)
            await self.save_state(state, run_id, agent_id)
            return AgentRunResult(
                status="cancelled",
                output="",
                total_iterations=state.iteration,
                run_id=run_id,
                error="Agent was cancelled",
                messages=state.messages,
            )

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            self._emit(EventType.RUN_ERROR, run_id, agent_id, state.iteration, error=str(e))
            await self._memory_on_run_error(input, str(e), state)
            await self.save_state(state, run_id, agent_id)
            return AgentRunResult(
                status="error",
                output="",
                total_iterations=state.iteration,
                total_llm_calls=state.total_llm_calls,
                total_tool_calls=state.total_tool_calls,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                run_id=run_id,
                error=str(e),
                messages=state.messages,
            )

        # Get output
        output = ""
        if hasattr(self.loop, 'synthesize'):
            output = await self.loop.synthesize(state)
        else:
            output = self.loop.get_output(state)

        # Save to memory via MemoryManager
        await self.save_to_memory(state, input, output)

        self._emit(EventType.RUN_COMPLETED, run_id, agent_id, state.iteration,
                    output_length=len(output))

        return AgentRunResult(
            status="completed",
            output=output,
            total_iterations=state.iteration,
            total_llm_calls=state.total_llm_calls,
            total_tool_calls=state.total_tool_calls,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            run_id=run_id,
            messages=state.messages,
        )

    async def run_with_state(
        self,
        state: AgentState,
        *,
        agent_id: str = "",
        run_id: str | None = None,
    ) -> AgentRunResult:
        """
        Run the agent loop with a pre-built state.

        For advanced use cases where you need full control over the
        initial state (custom messages, pre-injected memory, etc.).

        Args:
            state: Pre-built AgentState.
            agent_id: ID of the agent.
            run_id: Optional run ID (generated if not provided).

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        run_id = run_id or str(uuid.uuid4())

        await self._ensure_components_started()

        if hasattr(self.loop, 'run_id'):
            self.loop.run_id = run_id
        if hasattr(self.loop, 'agent_id'):
            self.loop.agent_id = agent_id

        self._emit(EventType.RUN_STARTED, run_id, agent_id)

        try:
            state = await self._execute_loop(state, run_id, agent_id)
        except asyncio.TimeoutError:
            self._emit(EventType.RUN_TIMEOUT, run_id, agent_id, state.iteration)
            return AgentRunResult(
                status="timeout", output="", total_iterations=state.iteration,
                run_id=run_id, error="Timed out", messages=state.messages,
            )
        except asyncio.CancelledError:
            self._emit(EventType.RUN_CANCELLED, run_id, agent_id, state.iteration)
            return AgentRunResult(
                status="cancelled", output="", total_iterations=state.iteration,
                run_id=run_id, error="Cancelled", messages=state.messages,
            )
        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            self._emit(EventType.RUN_ERROR, run_id, agent_id, state.iteration, error=str(e))
            return AgentRunResult(
                status="error", output="", total_iterations=state.iteration,
                run_id=run_id, error=str(e), messages=state.messages,
            )

        output = ""
        if hasattr(self.loop, 'synthesize'):
            output = await self.loop.synthesize(state)
        else:
            output = self.loop.get_output(state)

        self._emit(EventType.RUN_COMPLETED, run_id, agent_id, state.iteration)

        return AgentRunResult(
            status="completed",
            output=output,
            total_iterations=state.iteration,
            total_llm_calls=state.total_llm_calls,
            total_tool_calls=state.total_tool_calls,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            run_id=run_id,
            messages=state.messages,
        )

    async def stream(
        self,
        input: str,
        *,
        agent_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream agent execution events.

        Yields StreamEvent objects for real-time observation.

        Args:
            input: The user's input/objective.
            agent_id: ID of the agent.
            context: Optional additional context dict.

        Yields:
            StreamEvent objects.
        """
        run_id = str(uuid.uuid4())

        await self._ensure_components_started()

        state = self.create_state(input, context)
        await self.inject_memory_context(state, input)
        await self._memory_on_run_start(input, state)

        if hasattr(self.loop, 'run_id'):
            self.loop.run_id = run_id
        if hasattr(self.loop, 'agent_id'):
            self.loop.agent_id = agent_id

        while True:
            if state.is_cancelled:
                break

            yield StreamEvent(type="iteration_start", iteration=state.iteration + 1)

            async for event in self.loop.stream_step(state):
                yield event

            # Notify memory of iteration
            await self._memory_on_iteration(state, state.iteration)

            if not self.loop.should_continue(state):
                break

        # Synthesis for plan-critique
        if hasattr(self.loop, 'synthesize'):
            output = await self.loop.synthesize(state)
            yield StreamEvent(type="text_delta", text=output)

        yield StreamEvent(type="done", data={
            "total_iterations": state.iteration,
            "total_llm_calls": state.total_llm_calls,
            "total_tool_calls": state.total_tool_calls,
        })

    async def _execute_loop(self, state: AgentState, run_id: str, agent_id: str) -> AgentState:
        """Execute the agent loop until done, with optional per-iteration timeout."""
        while True:
            if state.is_cancelled:
                break

            self._emit(EventType.ITERATION_STARTED, run_id, agent_id, state.iteration + 1)

            if self.iteration_timeout:
                try:
                    state = await asyncio.wait_for(
                        self.loop.step(state),
                        timeout=self.iteration_timeout,
                    )
                except asyncio.TimeoutError:
                    self._emit(
                        EventType.ITERATION_COMPLETED, run_id, agent_id, state.iteration,
                        timeout=True, elapsed=state.elapsed_time,
                    )
                    raise
            else:
                state = await self.loop.step(state)

            self._emit(EventType.ITERATION_COMPLETED, run_id, agent_id, state.iteration)

            # Notify memory of iteration
            await self._memory_on_iteration(state, state.iteration)

            # Save state periodically
            if (self.state_store
                    and self.checkpoint_interval > 0
                    and state.iteration % self.checkpoint_interval == 0):
                await self.save_state(state, run_id, agent_id)

            if not self.loop.should_continue(state):
                break

        return state
