"""
Long-running task management for agent runs.

TaskManager supports:
- Background execution with asyncio
- Pause/resume via checkpointing (state saved on pause, restored on resume)
- Progress tracking and callbacks
- Task queuing and concurrency limits
- Integration with StateStore for fault tolerance

Deferred from Phase 2.4: Subagent spawn_background/get_result apply to subagents only.
This module provides general long-running task management for any agent run.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from curio_agent_sdk.core.events import (
    HookContext,
    HookRegistry,
    AGENT_ITERATION_AFTER,
)
from curio_agent_sdk.models.agent import AgentRunResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curio_agent_sdk.core.agent import Agent

logger = logging.getLogger(__name__)

# Task status values (aligned with typical run outcomes)
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"
TASK_PAUSED = "paused"
TASK_TIMEOUT = "timeout"


@dataclass
class TaskStatus:
    """Current status of a long-running task."""

    task_id: str
    status: str  # pending | running | completed | failed | cancelled | paused | timeout
    iteration: int = 0
    max_iterations: int = 0
    run_id: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskInfo:
    """Summary info for a task (for list_tasks)."""

    task_id: str
    agent_id: str
    input_preview: str  # truncated input for display
    status: str
    iteration: int
    max_iterations: int
    run_id: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None


def _truncate(s: str, max_len: int = 200) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


class TaskManager:
    """
    Manages long-running agent tasks with background execution, pause/resume,
    progress tracking, and optional concurrency limits.

    Uses the agent's StateStore for checkpointing: when a task is paused
    (cancelled), state is saved under run_id = task_id; resume loads that
    state and continues.

    Example:
        task_mgr = TaskManager(max_concurrent=2)
        task_id = await task_mgr.submit(agent, "Comprehensive analysis of X")
        status = await task_mgr.get_status(task_id)
        result = await task_mgr.wait(task_id, timeout=600)
    """

    def __init__(self, max_concurrent: int | None = None) -> None:
        """
        Args:
            max_concurrent: Max number of tasks running at once. None = no limit.
        """
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
        )
        # task_id -> TaskEntry
        self._tasks: dict[str, "_TaskEntry"] = {}
        self._lock = asyncio.Lock()
        # Progress: run_id (task_id) -> list of callbacks
        self._progress_callbacks: dict[str, list[Callable[..., Any]]] = {}
        # One-time hook registration per agent (id(agent)) so we don't double-register
        self._progress_hook_registered: set[int] = set()

    def _ensure_progress_hook(self, agent: "Agent") -> None:
        """Register a hook on the agent's runtime to dispatch progress by run_id."""
        aid = id(agent)
        if aid in self._progress_hook_registered:
            return
        registry: HookRegistry | None = getattr(
            getattr(agent, "runtime", None), "hook_registry", None
        )
        if registry is None:
            return
        task_mgr = self

        def _on_iteration(ctx: HookContext) -> None:
            run_id = ctx.run_id or ""
            if not run_id:
                return
            callbacks = task_mgr._progress_callbacks.get(run_id, [])
            iteration = ctx.iteration
            max_iterations = getattr(ctx.state, "max_iterations", 0) if ctx.state else 0
            for cb in callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        asyncio.create_task(
                            cb(run_id, iteration, max_iterations)
                        )
                    else:
                        cb(run_id, iteration, max_iterations)
                except Exception as e:
                    logger.warning("Progress callback failed: %s", e)
            # Update stored iteration for get_status (same object, no lock needed for these assigns)
            entry = task_mgr._tasks.get(run_id)
            if entry is not None:
                entry.iteration = iteration
                entry.max_iterations = max_iterations
                entry.updated_at = datetime.utcnow()

        # Use a wrapper that captures context correctly
        def _handler(ctx: HookContext) -> None:
            _on_iteration(ctx)

        registry.on(AGENT_ITERATION_AFTER, _handler, priority=100)
        self._progress_hook_registered.add(aid)

    async def submit(
        self,
        agent: "Agent",
        input_text: str,
        **kwargs: Any,
    ) -> str:
        """
        Submit a task to run in the background. Returns task_id.

        The task uses run_id=task_id so checkpoints are stored under task_id
        and resume_from=task_id works for pause/resume.

        Args:
            agent: Agent to run.
            input_text: User input / objective for the run.
            **kwargs: Passed through to agent.arun() (context, max_iterations, timeout, etc.).

        Returns:
            task_id: Use get_status(task_id), get_result(task_id), cancel(task_id), etc.
        """
        task_id = str(uuid.uuid4())
        run_id = task_id

        async def _run() -> None:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            if self._semaphore is not None:
                async with self._semaphore:
                    await self._run_one(task_id, agent, input_text, kwargs, run_id)
            else:
                await self._run_one(task_id, agent, input_text, kwargs, run_id)

        async with self._lock:
            entry = _TaskEntry(
                task_id=task_id,
                run_id=run_id,
                agent=agent,
                input_text=input_text,
                kwargs=kwargs,
                status=TASK_PENDING,
                iteration=0,
                max_iterations=kwargs.get("max_iterations") or getattr(agent, "max_iterations", 25),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            self._tasks[task_id] = entry

        self._ensure_progress_hook(agent)
        entry.status = TASK_RUNNING
        entry.updated_at = datetime.utcnow()
        entry.task = asyncio.create_task(_run())
        return task_id

    async def _run_one(
        self,
        task_id: str,
        agent: "Agent",
        input_text: str,
        kwargs: dict[str, Any],
        run_id: str,
    ) -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return
        try:
            result = await agent.arun(
                input_text,
                run_id=run_id,
                **{k: v for k, v in kwargs.items() if k != "run_id"},
            )
            async with self._lock:
                if task_id not in self._tasks:
                    return
                e = self._tasks[task_id]
                e.result = result
                e.updated_at = datetime.utcnow()
                if result.status == "completed":
                    e.status = TASK_COMPLETED
                elif result.status == "timeout":
                    e.status = TASK_TIMEOUT
                elif result.status == "cancelled":
                    e.status = TASK_PAUSED  # Cancelled by pause() -> treat as paused
                else:
                    e.status = TASK_FAILED
                    e.error = getattr(result, "error", None) or ""
        except asyncio.CancelledError:
            async with self._lock:
                if task_id in self._tasks:
                    e = self._tasks[task_id]
                    if e.status == TASK_RUNNING:
                        e.status = TASK_PAUSED
                    e.updated_at = datetime.utcnow()
        except Exception as ex:
            logger.exception("Task %s failed: %s", task_id, ex)
            async with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = TASK_FAILED
                    self._tasks[task_id].error = str(ex)
                    self._tasks[task_id].updated_at = datetime.utcnow()
                    self._tasks[task_id].result = AgentRunResult(
                        status="error",
                        output="",
                        error=str(ex),
                        run_id=run_id,
                    )
        finally:
            self._progress_callbacks.pop(task_id, None)

    async def get_status(self, task_id: str) -> TaskStatus | None:
        """Return current status for the task, or None if unknown."""
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return None
            return TaskStatus(
                task_id=entry.task_id,
                status=entry.status,
                iteration=entry.iteration,
                max_iterations=entry.max_iterations,
                run_id=entry.run_id,
                error=entry.error,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
            )

    async def get_result(self, task_id: str) -> AgentRunResult | None:
        """
        Return the result of the task if it has completed (completed, failed, timeout, paused).
        Returns None if the task is still running or unknown.
        """
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return None
            if entry.status in (TASK_COMPLETED, TASK_FAILED, TASK_TIMEOUT, TASK_PAUSED, TASK_CANCELLED):
                return entry.result
            return None

    async def cancel(self, task_id: str) -> None:
        """Cancel the task. If the agent uses StateStore, state is saved on cancel (pause)."""
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
                entry.status = TASK_CANCELLED
                entry.updated_at = datetime.utcnow()

    async def pause(self, task_id: str) -> None:
        """
        Pause the task by cancelling the current run. State is checkpointed on cancel
        (Runtime saves state in CancelledError handler). Use resume(task_id) to continue.
        """
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
            entry.status = TASK_PAUSED
            entry.updated_at = datetime.utcnow()

    async def resume(self, task_id: str) -> None:
        """
        Resume a paused task by loading state from StateStore (run_id=task_id) and
        running the agent again with resume_from=task_id.
        """
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise ValueError(f"Unknown task: {task_id}")
            if entry.status != TASK_PAUSED:
                raise ValueError(f"Task {task_id} is not paused (status={entry.status})")
            agent = entry.agent
            input_text = entry.input_text
            kwargs = dict(entry.kwargs)

        run_id = task_id
        async def _run_resumed() -> None:
            if self._semaphore is not None:
                async with self._semaphore:
                    await self._run_one_resumed(task_id, agent, input_text, kwargs, run_id)
            else:
                await self._run_one_resumed(task_id, agent, input_text, kwargs, run_id)

        async with self._lock:
            entry.status = TASK_RUNNING
            entry.updated_at = datetime.utcnow()
            entry.result = None
            entry.error = None
            entry.task = asyncio.create_task(_run_resumed())

    async def _run_one_resumed(
        self,
        task_id: str,
        agent: "Agent",
        input_text: str,
        kwargs: dict[str, Any],
        run_id: str,
    ) -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return
        try:
            result = await agent.arun(
                input_text,
                resume_from=run_id,
                run_id=run_id,
                **{k: v for k, v in kwargs.items() if k not in ("run_id", "resume_from")},
            )
            async with self._lock:
                if task_id not in self._tasks:
                    return
                e = self._tasks[task_id]
                e.result = result
                e.updated_at = datetime.utcnow()
                if result.status == "completed":
                    e.status = TASK_COMPLETED
                elif result.status == "timeout":
                    e.status = TASK_TIMEOUT
                elif result.status == "cancelled":
                    e.status = TASK_PAUSED
                else:
                    e.status = TASK_FAILED
                    e.error = getattr(result, "error", None) or ""
        except asyncio.CancelledError:
            async with self._lock:
                if task_id in self._tasks:
                    e = self._tasks[task_id]
                    if e.status == TASK_RUNNING:
                        e.status = TASK_PAUSED
                    e.updated_at = datetime.utcnow()
        except Exception as ex:
            logger.exception("Task %s resume failed: %s", task_id, ex)
            async with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = TASK_FAILED
                    self._tasks[task_id].error = str(ex)
                    self._tasks[task_id].updated_at = datetime.utcnow()
                    self._tasks[task_id].result = AgentRunResult(
                        status="error",
                        output="",
                        error=str(ex),
                        run_id=run_id,
                    )
        finally:
            self._progress_callbacks.pop(task_id, None)

    def on_progress(
        self,
        task_id: str,
        callback: Callable[[str, int, int], Any] | Callable[[str, int, int], Awaitable[Any]],
    ) -> None:
        """
        Register a callback to be invoked on each iteration for this task.

        Callback receives (run_id, iteration, max_iterations).
        """
        if task_id not in self._progress_callbacks:
            self._progress_callbacks[task_id] = []
        self._progress_callbacks[task_id].append(callback)

    async def wait(self, task_id: str, timeout: float | None = None) -> AgentRunResult | None:
        """
        Wait for the task to complete (or timeout). Returns the result when done.

        Args:
            task_id: Task ID from submit().
            timeout: Max seconds to wait. None = wait forever.

        Returns:
            AgentRunResult when the task completes (or after timeout). None if task unknown.
        """
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return None
            task = entry.task

        if task is None:
            return await self.get_result(task_id)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            return await self.get_result(task_id)
        except asyncio.CancelledError:
            raise
        return await self.get_result(task_id)

    async def scan_incomplete(
        self,
        agent: "Agent",
    ) -> list[RecoveredRun]:
        """
        Scan the agent's StateStore for incomplete runs that may need recovery.

        Returns a list of RecoveredRun objects describing runs that were
        interrupted (e.g., by a crash) and can be resumed.

        Args:
            agent: Agent whose StateStore to scan.

        Returns:
            List of RecoveredRun objects for runs that have saved state
            but are not tracked by this TaskManager (i.e., were interrupted).
        """
        state_store = getattr(getattr(agent, "runtime", None), "state_store", None)
        if state_store is None:
            return []

        agent_id = getattr(agent, "agent_id", "")
        try:
            run_ids = await state_store.list_runs(agent_id)
        except Exception as e:
            logger.warning("Failed to list runs for recovery: %s", e)
            return []

        recovered: list[RecoveredRun] = []
        for run_id in run_ids:
            # Skip runs that are already tracked in this TaskManager
            if run_id in self._tasks:
                continue
            try:
                state = await state_store.load(run_id)
                if state is None:
                    continue
                # A run with saved state that is not done is considered incomplete
                if not state.done:
                    recovered.append(RecoveredRun(
                        run_id=run_id,
                        agent_id=agent_id,
                        iteration=state.iteration,
                        max_iterations=state.max_iterations,
                    ))
            except Exception as e:
                logger.warning("Failed to load state for run %s: %s", run_id, e)
        return recovered

    async def recover_incomplete(
        self,
        agent: "Agent",
        input_text: str = "",
        **kwargs: Any,
    ) -> list[str]:
        """
        Scan StateStore for incomplete runs and resume them.

        Each incomplete run is submitted as a resumed task. Returns the
        list of task IDs for the recovered runs.

        Args:
            agent: Agent to resume runs for.
            input_text: Fallback input text for resumed runs.
            **kwargs: Passed to agent.arun() on resume.

        Returns:
            List of task_id strings for recovered runs.
        """
        incomplete = await self.scan_incomplete(agent)
        task_ids: list[str] = []
        for run_info in incomplete:
            task_id = run_info.run_id
            run_id = task_id
            entry = _TaskEntry(
                task_id=task_id,
                run_id=run_id,
                agent=agent,
                input_text=input_text,
                kwargs=dict(kwargs),
                status=TASK_PAUSED,
                iteration=run_info.iteration,
                max_iterations=run_info.max_iterations,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            async with self._lock:
                self._tasks[task_id] = entry
            try:
                await self.resume(task_id)
                task_ids.append(task_id)
                logger.info("Recovered incomplete run %s at iteration %d", run_id, run_info.iteration)
            except Exception as e:
                logger.warning("Failed to recover run %s: %s", run_id, e)
        return task_ids

    async def list_tasks(self, status: str | None = None) -> list[TaskInfo]:
        """
        List tasks, optionally filtered by status.

        Args:
            status: If set, only return tasks with this status (e.g. "running", "paused").
        """
        async with self._lock:
            items = list(self._tasks.values())
        if status is not None:
            items = [e for e in items if e.status == status]
        return [
            TaskInfo(
                task_id=e.task_id,
                agent_id=getattr(e.agent, "agent_id", ""),
                input_preview=_truncate(e.input_text),
                status=e.status,
                iteration=e.iteration,
                max_iterations=e.max_iterations,
                run_id=e.run_id,
                created_at=e.created_at,
                updated_at=e.updated_at,
                error=e.error,
            )
            for e in items
        ]


@dataclass
class RecoveredRun:
    """Information about an incomplete run found during crash recovery."""

    run_id: str
    agent_id: str
    iteration: int
    max_iterations: int


@dataclass
class _TaskEntry:
    """Internal entry for a single task."""

    task_id: str
    run_id: str
    agent: "Agent"
    input_text: str
    kwargs: dict[str, Any]
    status: str
    iteration: int
    max_iterations: int
    created_at: datetime
    updated_at: datetime
    task: asyncio.Task | None = None
    result: AgentRunResult | None = None
    error: str | None = None
