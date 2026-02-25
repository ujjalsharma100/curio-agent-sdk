"""
The main Agent class - composes a loop, tools, LLM client, middleware,
memory, and checkpointing into a complete autonomous agent.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from curio_agent_sdk.core.context import ContextManager
from curio_agent_sdk.core.loops.base import AgentLoop
from curio_agent_sdk.core.loops.tool_calling import ToolCallingLoop
from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.tools.executor import ToolExecutor
from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.models.llm import Message
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.models.events import AgentEvent, EventType, StreamEvent
from curio_agent_sdk.exceptions import AgentTimeoutError, AgentCancelledError, MaxIterationsError

if TYPE_CHECKING:
    from curio_agent_sdk.memory.base import Memory
    from curio_agent_sdk.core.checkpoint import CheckpointStore

logger = logging.getLogger(__name__)


class Agent:
    """
    The primary agent class. Composes:
    - A loop pattern (ToolCallingLoop, PlanCritiqueSynthesizeLoop, or custom)
    - Tools that the agent can use
    - An LLM client for making model calls
    - A system prompt defining the agent's persona
    - Optional middleware pipeline for observability and control
    - Optional context manager for token budget management
    - Optional human-in-the-loop handler for tool confirmation
    - Optional memory for cross-turn/cross-session knowledge
    - Optional checkpoint store for resumable execution

    Simple usage:
        agent = Agent(
            model="openai:gpt-4o",
            tools=[search, calculator],
            system_prompt="You are a helpful assistant.",
        )
        result = agent.run("What is the weather in SF?")

    Full configuration:
        from curio_agent_sdk.middleware import LoggingMiddleware, CostTracker
        from curio_agent_sdk.memory import ConversationMemory, VectorMemory, CompositeMemory
        from curio_agent_sdk.core.checkpoint import FileCheckpointStore

        agent = Agent(
            loop=ToolCallingLoop(tier="tier3"),
            llm=LLMClient(router=my_router),
            tools=[search, calculator],
            system_prompt="You are a research agent.",
            agent_id="research-agent",
            max_iterations=25,
            timeout=300,
            iteration_timeout=60,
            context_manager=ContextManager(max_tokens=128000),
            middleware=[LoggingMiddleware(), CostTracker(budget=1.0)],
            memory=CompositeMemory({
                "conversation": ConversationMemory(max_entries=50),
                "knowledge": VectorMemory(),
            }),
            checkpoint_store=FileCheckpointStore("./checkpoints"),
        )
        result = await agent.arun("Research quantum computing advances")
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
        timeout: float | None = None,  # Total run timeout in seconds
        iteration_timeout: float | None = None,  # Per-step timeout in seconds
        max_tokens: int = 4096,
        temperature: float = 0.7,

        # Context management
        context_manager: ContextManager | None = None,

        # Middleware
        middleware: list | None = None,  # list[Middleware]

        # Human-in-the-loop
        human_input: Any | None = None,  # HumanInputHandler

        # Memory
        memory: Memory | None = None,

        # Checkpointing
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_interval: int = 1,  # Save checkpoint every N iterations

        # Callbacks
        on_event: Callable[[AgentEvent], None] | None = None,
    ):
        self.system_prompt = system_prompt
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.iteration_timeout = iteration_timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.on_event = on_event
        self.context_manager = context_manager
        self.middleware = middleware or []
        self.human_input = human_input
        self.memory = memory
        self.checkpoint_store = checkpoint_store
        self.checkpoint_interval = checkpoint_interval

        # Set up tools
        self.registry = ToolRegistry()
        for t in (tools or []):
            self.registry.register(t)
        self.executor = ToolExecutor(self.registry, human_input=human_input)

        # Set up LLM client
        if llm:
            self.llm = llm
        elif model:
            # Parse "provider:model" shorthand
            provider, _, model_name = model.partition(":")
            from curio_agent_sdk.llm.router import TieredRouter
            router = TieredRouter()
            if provider and model_name:
                # Override the default tier with the specified model
                from curio_agent_sdk.llm.router import ModelPriority, TierConfig
                router.tiers[tier] = TierConfig(
                    name=tier,
                    model_priority=[ModelPriority(provider=provider, model=model_name)],
                )
            self.llm = LLMClient(router=router)
        else:
            self.llm = LLMClient()

        # Wrap LLM client with middleware if provided
        if self.middleware:
            from curio_agent_sdk.middleware.base import MiddlewarePipeline
            self._middleware_pipeline = MiddlewarePipeline(self.middleware)
            self.llm = self._middleware_pipeline.wrap_llm_client(self.llm)
        else:
            self._middleware_pipeline = None

        # Set up loop
        if loop:
            self.loop = loop
        else:
            self.loop = ToolCallingLoop(tier=tier, temperature=temperature, max_tokens=max_tokens)

        # Wire up the loop with LLM, executor, and context manager
        self._wire_loop()

    def _wire_loop(self):
        """Inject LLM client, tool executor, and context manager into the loop."""
        if hasattr(self.loop, 'llm') and self.loop.llm is None:
            self.loop.llm = self.llm
        if hasattr(self.loop, 'tool_executor') and self.loop.tool_executor is None:
            self.loop.tool_executor = self.executor
        if self.context_manager is not None:
            self.loop.context_manager = self.context_manager

    def _create_state(self, input_text: str, context: dict[str, Any] | None = None) -> AgentState:
        """Create initial agent state for a run."""
        messages = [Message.system(self.system_prompt)]

        # Add context as part of the user message if provided
        if context:
            import json
            input_with_context = f"{input_text}\n\nAdditional context:\n{json.dumps(context, indent=2)}"
            messages.append(Message.user(input_with_context))
        else:
            messages.append(Message.user(input_text))

        return AgentState(
            messages=messages,
            tools=self.registry.tools,
            tool_schemas=self.registry.get_llm_schemas(),
            max_iterations=self.max_iterations,
        )

    async def _inject_memory_context(self, state: AgentState, input_text: str):
        """Inject memory context into the system prompt if memory is available."""
        if self.memory is None:
            return

        try:
            memory_context = await self.memory.get_context(input_text, max_tokens=2000)
            if memory_context:
                # Insert memory context as a system message after the main system prompt
                state.messages.insert(1, Message.system(
                    f"Relevant information from memory:\n{memory_context}"
                ))
        except Exception as e:
            logger.warning("Failed to retrieve memory context: %s", e)

    async def _save_to_memory(self, state: AgentState, input_text: str, output: str):
        """Save the interaction to memory after a successful run."""
        if self.memory is None:
            return

        try:
            await self.memory.add(
                f"User: {input_text}",
                metadata={"type": "user_input", "role": "user"},
            )
            if output:
                await self.memory.add(
                    f"Assistant: {output}",
                    metadata={"type": "assistant_output", "role": "assistant"},
                )
        except Exception as e:
            logger.warning("Failed to save to memory: %s", e)

    async def _save_checkpoint(self, state: AgentState, run_id: str):
        """Save a checkpoint of the current state."""
        if self.checkpoint_store is None:
            return

        try:
            from curio_agent_sdk.core.checkpoint import Checkpoint
            checkpoint = Checkpoint.from_state(state, run_id=run_id, agent_id=self.agent_id)
            await self.checkpoint_store.save(checkpoint)
            self._emit(EventType.CHECKPOINT_SAVED, run_id, state.iteration)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    async def _restore_from_checkpoint(self, run_id: str) -> AgentState | None:
        """Restore agent state from a checkpoint."""
        if self.checkpoint_store is None:
            return None

        try:
            checkpoint = await self.checkpoint_store.load(run_id)
            if checkpoint is None:
                return None

            messages = checkpoint.restore_messages()
            state = AgentState(
                messages=messages,
                tools=self.registry.tools,
                tool_schemas=self.registry.get_llm_schemas(),
                iteration=checkpoint.iteration,
                max_iterations=self.max_iterations,
                metadata=checkpoint.metadata,
                total_llm_calls=checkpoint.total_llm_calls,
                total_tool_calls=checkpoint.total_tool_calls,
                total_input_tokens=checkpoint.total_input_tokens,
                total_output_tokens=checkpoint.total_output_tokens,
            )

            self._emit(EventType.CHECKPOINT_RESTORED, run_id, state.iteration)
            logger.info("Restored checkpoint for run %s at iteration %d", run_id, state.iteration)
            return state

        except Exception as e:
            logger.warning("Failed to restore checkpoint for run %s: %s", run_id, e)
            return None

    def _emit(self, event_type: EventType, run_id: str, iteration: int = 0, **data):
        """Emit an agent event."""
        if self.on_event:
            try:
                event = AgentEvent(
                    type=event_type,
                    run_id=run_id,
                    agent_id=self.agent_id,
                    iteration=iteration,
                    data=data,
                )
                self.on_event(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

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
            resume_from: Optional run_id to resume from a checkpoint.

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        run_id = resume_from or str(uuid.uuid4())
        effective_timeout = timeout or self.timeout

        # Try to resume from checkpoint
        state = None
        if resume_from:
            state = await self._restore_from_checkpoint(resume_from)

        # Create fresh state if not resuming
        if state is None:
            state = self._create_state(input, context)
            # Inject memory context for fresh runs
            await self._inject_memory_context(state, input)

        if max_iterations:
            state.max_iterations = max_iterations

        # Set run_id on loop if supported
        if hasattr(self.loop, 'run_id'):
            self.loop.run_id = run_id
        if hasattr(self.loop, 'agent_id'):
            self.loop.agent_id = self.agent_id

        self._emit(EventType.RUN_STARTED, run_id, data={
            "input": input,
            "resumed": resume_from is not None,
        })

        try:
            if effective_timeout:
                state = await asyncio.wait_for(
                    self._execute_loop(state, run_id),
                    timeout=effective_timeout,
                )
            else:
                state = await self._execute_loop(state, run_id)

        except asyncio.TimeoutError:
            self._emit(EventType.RUN_TIMEOUT, run_id, state.iteration,
                       elapsed=state.elapsed_time)
            await self._save_checkpoint(state, run_id)
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
            self._emit(EventType.RUN_CANCELLED, run_id, state.iteration,
                       elapsed=state.elapsed_time)
            await self._save_checkpoint(state, run_id)
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
            self._emit(EventType.RUN_ERROR, run_id, state.iteration, error=str(e))
            await self._save_checkpoint(state, run_id)
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
        # Check if loop has a synthesize step (plan-critique pattern)
        if hasattr(self.loop, 'synthesize'):
            output = await self.loop.synthesize(state)
        else:
            output = self.loop.get_output(state)

        # Save to memory
        await self._save_to_memory(state, input, output)

        self._emit(EventType.RUN_COMPLETED, run_id, state.iteration,
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

    async def _execute_loop(self, state: AgentState, run_id: str) -> AgentState:
        """Execute the agent loop until done, with optional per-iteration timeout."""
        while True:
            if state.is_cancelled:
                break

            self._emit(EventType.ITERATION_STARTED, run_id, state.iteration + 1)

            if self.iteration_timeout:
                try:
                    state = await asyncio.wait_for(
                        self.loop.step(state),
                        timeout=self.iteration_timeout,
                    )
                except asyncio.TimeoutError:
                    self._emit(
                        EventType.ITERATION_COMPLETED, run_id, state.iteration,
                        timeout=True, elapsed=state.elapsed_time,
                    )
                    raise
            else:
                state = await self.loop.step(state)

            self._emit(EventType.ITERATION_COMPLETED, run_id, state.iteration)

            # Save checkpoint periodically
            if (self.checkpoint_store
                    and self.checkpoint_interval > 0
                    and state.iteration % self.checkpoint_interval == 0):
                await self._save_checkpoint(state, run_id)

            if not self.loop.should_continue(state):
                break

        return state

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
        run_id = str(uuid.uuid4())
        state = self._create_state(input, context)

        # Inject memory context
        await self._inject_memory_context(state, input)

        if hasattr(self.loop, 'run_id'):
            self.loop.run_id = run_id
        if hasattr(self.loop, 'agent_id'):
            self.loop.agent_id = self.agent_id

        while True:
            if state.is_cancelled:
                break

            yield StreamEvent(type="iteration_start", iteration=state.iteration + 1)

            async for event in self.loop.stream_step(state):
                yield event

            # Update state after streaming step (step modifies state in-place during stream)
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
            resume_from: Optional run_id to resume from a checkpoint.

        Returns:
            AgentRunResult with status, output, and metrics.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context - can't use asyncio.run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.arun(input, context, max_iterations, timeout, resume_from),
                )
                return future.result()
        except RuntimeError:
            # No running event loop - use asyncio.run directly
            return asyncio.run(self.arun(input, context, max_iterations, timeout, resume_from))
