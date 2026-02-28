"""
Distributed event bus for cross-process / cross-machine agent event streaming.

The EventBus extends the in-process HookRegistry with pattern-based pub/sub,
event replay, dead-letter handling, and pluggable backends (in-memory, Redis, Kafka).

Design:
- EventBus is an ABC; backends implement publish/subscribe/replay.
- EventBusBridge is a Component that auto-publishes HookRegistry events to the bus.
- EventFilter supports glob-pattern matching on dotted event names.
- InMemoryEventBus is the default for single-process use.
- Redis/Kafka backends are optional dependencies for distributed deployments.

Integration:
- Pass an EventBus to Agent/Builder via .event_bus(bus).
- Runtime creates an EventBusBridge that connects HookRegistry → EventBus.
- External subscribers receive AgentEvent objects serialised via to_dict().

Resolves REVIEW §5.4 (event system is read-only / no cross-process fan-out).
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Awaitable, TYPE_CHECKING

from curio_agent_sdk.base import Component
from curio_agent_sdk.models.events import AgentEvent, EventType

if TYPE_CHECKING:
    from curio_agent_sdk.core.hooks import HookRegistry, HookContext

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────

EventHandler = Callable[[AgentEvent], Any] | Callable[[AgentEvent], Awaitable[Any]]


def _is_async(fn: EventHandler) -> bool:
    if asyncio.iscoroutinefunction(fn):
        return True
    if callable(fn) and hasattr(fn, "__call__"):
        return asyncio.iscoroutinefunction(getattr(fn, "__call__", None))
    return False


# ── Event filter ─────────────────────────────────────────────────────────

class EventFilter:
    """
    Pattern-based event filter using glob/fnmatch syntax on dotted event names.

    Patterns:
        "*"              — matches everything
        "agent.*"        — matches agent.run.before, agent.iteration.after, etc.
        "tool.call.*"    — matches tool.call.before, tool.call.after, tool.call.error
        "llm.call.after" — exact match
        "*.error"        — matches agent.run.error, llm.call.error, tool.call.error
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def matches(self, event_name: str) -> bool:
        """Return True if event_name matches this filter's pattern."""
        return fnmatch.fnmatch(event_name, self.pattern)

    def __repr__(self) -> str:
        return f"EventFilter({self.pattern!r})"


# ── Dead letter entry ────────────────────────────────────────────────────

@dataclass
class DeadLetterEntry:
    """A failed event delivery stored for inspection / retry."""
    event: AgentEvent
    handler: str  # repr of the handler
    error: str
    timestamp: float = field(default_factory=time.time)
    pattern: str = ""


# ── EventBus ABC ─────────────────────────────────────────────────────────

class EventBus(ABC):
    """
    Abstract base for distributed event buses.

    Backends implement publish, subscribe, unsubscribe, and optionally replay.
    All backends must also implement the Component lifecycle (startup/shutdown).
    """

    @abstractmethod
    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to all matching subscribers."""
        ...

    @abstractmethod
    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> None:
        """
        Subscribe a handler to events matching a glob pattern.

        Pattern examples: "*", "agent.*", "tool.call.*", "llm.call.after"
        """
        ...

    @abstractmethod
    async def unsubscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> None:
        """Remove a previously registered subscription."""
        ...

    async def replay(
        self,
        from_timestamp: float,
        pattern: str = "*",
    ) -> AsyncIterator[AgentEvent]:
        """
        Replay stored events from a given timestamp, filtered by pattern.

        Not all backends support replay. The default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support event replay"
        )
        # Make this an async generator so callers can iterate.
        yield  # type: ignore[misc]  # pragma: no cover

    @property
    def dead_letters(self) -> list[DeadLetterEntry]:
        """Access the dead letter queue (failed deliveries). Override in backends."""
        return []


# ── InMemoryEventBus ─────────────────────────────────────────────────────

class InMemoryEventBus(EventBus, Component):
    """
    Single-process event bus with full replay support and dead letter queue.

    Good for:
    - Testing and development
    - Single-process multi-agent setups
    - Prototyping before switching to Redis/Kafka

    Stores all published events in a bounded deque for replay.
    """

    def __init__(
        self,
        *,
        max_history: int = 10_000,
        max_dead_letters: int = 1_000,
    ) -> None:
        # pattern -> list of handlers
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        # Event history for replay (bounded)
        self._history: deque[AgentEvent] = deque(maxlen=max_history)
        # Dead letter queue
        self._dead_letters: deque[DeadLetterEntry] = deque(maxlen=max_dead_letters)
        self._started = False

    # ── Component lifecycle ──────────────────────────────────────────

    async def startup(self) -> None:
        self._started = True
        logger.debug("InMemoryEventBus started")

    async def shutdown(self) -> None:
        self._subscribers.clear()
        self._started = False
        logger.debug("InMemoryEventBus shut down")

    async def health_check(self) -> bool:
        return self._started

    # ── EventBus interface ───────────────────────────────────────────

    async def publish(self, event: AgentEvent) -> None:
        """Publish to all subscribers whose pattern matches the event type."""
        self._history.append(event)
        event_name = _event_type_to_hook_name(event.type)

        for pattern, handlers in self._subscribers.items():
            filt = EventFilter(pattern)
            if not filt.matches(event_name):
                continue
            for handler in handlers:
                try:
                    if _is_async(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.warning(
                        "EventBus handler %r failed for %s: %s",
                        handler, event_name, e,
                    )
                    self._dead_letters.append(DeadLetterEntry(
                        event=event,
                        handler=repr(handler),
                        error=str(e),
                        pattern=pattern,
                    ))

    async def subscribe(self, pattern: str, handler: EventHandler) -> None:
        self._subscribers[pattern].append(handler)

    async def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        if pattern in self._subscribers:
            self._subscribers[pattern] = [
                h for h in self._subscribers[pattern] if h is not handler
            ]
            if not self._subscribers[pattern]:
                del self._subscribers[pattern]

    async def replay(
        self,
        from_timestamp: float,
        pattern: str = "*",
    ) -> AsyncIterator[AgentEvent]:
        """Yield stored events from from_timestamp that match pattern."""
        filt = EventFilter(pattern)
        for event in self._history:
            if event.timestamp.timestamp() < from_timestamp:
                continue
            event_name = _event_type_to_hook_name(event.type)
            if filt.matches(event_name):
                yield event

    @property
    def dead_letters(self) -> list[DeadLetterEntry]:
        return list(self._dead_letters)

    def clear_dead_letters(self) -> None:
        """Clear the dead letter queue."""
        self._dead_letters.clear()

    def clear_history(self) -> None:
        """Clear the event history."""
        self._history.clear()


# ── EventBusBridge ───────────────────────────────────────────────────────

class EventBusBridge(Component):
    """
    Bridges HookRegistry lifecycle events into an EventBus.

    On startup, registers a handler on ALL known hook events that converts
    HookContext → AgentEvent and publishes to the bus. On shutdown, removes
    the handlers.

    This is the glue between the in-process hook system and the distributed
    event bus. It follows the same adapter pattern as _register_on_event_adapter
    in core/runtime.py but targets the EventBus instead of a single callback.
    """

    def __init__(
        self,
        bus: EventBus,
        hook_registry: HookRegistry,
    ) -> None:
        self.bus = bus
        self.hook_registry = hook_registry
        self._handler: Callable | None = None

    async def startup(self) -> None:
        """Register the forwarding hook and start the bus (if Component)."""
        # Start the bus itself if it has a lifecycle
        if isinstance(self.bus, Component):
            await self.bus.startup()

        # Register a single hook handler that forwards all events to the bus
        from curio_agent_sdk.core.hooks import HOOK_EVENTS

        async def _forward_to_bus(ctx: HookContext) -> None:
            event_type = _hook_name_to_event_type(ctx.event)
            if event_type is None:
                return
            agent_event = AgentEvent(
                type=event_type,
                run_id=ctx.run_id,
                agent_id=ctx.agent_id,
                iteration=ctx.iteration,
                data=dict(ctx.data),
            )
            await self.bus.publish(agent_event)

        self._handler = _forward_to_bus
        for ev in HOOK_EVENTS:
            self.hook_registry.on(ev, self._handler, priority=999)

        logger.debug("EventBusBridge started — forwarding %d hook events", len(HOOK_EVENTS))

    async def shutdown(self) -> None:
        """Remove the forwarding hook and shut down the bus."""
        if self._handler is not None:
            from curio_agent_sdk.core.hooks import HOOK_EVENTS
            for ev in HOOK_EVENTS:
                self.hook_registry.off(ev, self._handler)
            self._handler = None

        if isinstance(self.bus, Component):
            await self.bus.shutdown()

        logger.debug("EventBusBridge shut down")

    async def health_check(self) -> bool:
        if isinstance(self.bus, Component):
            return await self.bus.health_check()
        return True


# ── Mapping helpers ──────────────────────────────────────────────────────

# Hook name (dotted) → EventType mapping
_HOOK_TO_EVENT_TYPE: dict[str, EventType | Callable[[dict], EventType]] = {}


def _build_hook_to_event_map() -> None:
    """Build the mapping lazily on first use."""
    if _HOOK_TO_EVENT_TYPE:
        return
    from curio_agent_sdk.core.hooks import (
        AGENT_RUN_BEFORE, AGENT_RUN_AFTER, AGENT_RUN_ERROR,
        AGENT_ITERATION_BEFORE, AGENT_ITERATION_AFTER,
        LLM_CALL_BEFORE, LLM_CALL_AFTER, LLM_CALL_ERROR,
        TOOL_CALL_BEFORE, TOOL_CALL_AFTER, TOOL_CALL_ERROR,
        MEMORY_INJECT_BEFORE, MEMORY_SAVE_BEFORE, MEMORY_QUERY_BEFORE,
        STATE_CHECKPOINT_BEFORE, STATE_CHECKPOINT_AFTER,
    )
    _HOOK_TO_EVENT_TYPE.update({
        AGENT_RUN_BEFORE: EventType.RUN_STARTED,
        AGENT_RUN_AFTER: EventType.RUN_COMPLETED,
        AGENT_ITERATION_BEFORE: EventType.ITERATION_STARTED,
        AGENT_ITERATION_AFTER: EventType.ITERATION_COMPLETED,
        LLM_CALL_BEFORE: EventType.LLM_CALL_STARTED,
        LLM_CALL_AFTER: EventType.LLM_CALL_COMPLETED,
        LLM_CALL_ERROR: EventType.LLM_CALL_ERROR,
        TOOL_CALL_BEFORE: EventType.TOOL_CALL_STARTED,
        TOOL_CALL_AFTER: EventType.TOOL_CALL_COMPLETED,
        TOOL_CALL_ERROR: EventType.TOOL_CALL_ERROR,
        MEMORY_INJECT_BEFORE: EventType.CUSTOM,
        MEMORY_SAVE_BEFORE: EventType.CUSTOM,
        MEMORY_QUERY_BEFORE: EventType.CUSTOM,
        STATE_CHECKPOINT_BEFORE: EventType.CHECKPOINT_SAVED,
        STATE_CHECKPOINT_AFTER: EventType.CHECKPOINT_SAVED,
        # AGENT_RUN_ERROR is special — needs data inspection
    })


def _hook_name_to_event_type(hook_name: str, data: dict | None = None) -> EventType | None:
    """Convert a dotted hook event name to an EventType."""
    _build_hook_to_event_map()
    from curio_agent_sdk.core.hooks import AGENT_RUN_ERROR, STATE_CHECKPOINT_AFTER

    if hook_name == AGENT_RUN_ERROR:
        kind = (data or {}).get("error_kind", "error")
        if kind == "timeout":
            return EventType.RUN_TIMEOUT
        elif kind == "cancelled":
            return EventType.RUN_CANCELLED
        return EventType.RUN_ERROR

    if hook_name == STATE_CHECKPOINT_AFTER:
        action = (data or {}).get("checkpoint_action", "save")
        return EventType.CHECKPOINT_SAVED if action == "save" else EventType.CHECKPOINT_RESTORED

    return _HOOK_TO_EVENT_TYPE.get(hook_name)


# Reverse: EventType → canonical hook-style dotted name (for pattern matching)
_EVENT_TYPE_TO_HOOK_NAME: dict[EventType, str] = {
    EventType.RUN_STARTED: "agent.run.before",
    EventType.RUN_COMPLETED: "agent.run.after",
    EventType.RUN_ERROR: "agent.run.error",
    EventType.RUN_TIMEOUT: "agent.run.error",
    EventType.RUN_CANCELLED: "agent.run.error",
    EventType.ITERATION_STARTED: "agent.iteration.before",
    EventType.ITERATION_COMPLETED: "agent.iteration.after",
    EventType.LLM_CALL_STARTED: "llm.call.before",
    EventType.LLM_CALL_COMPLETED: "llm.call.after",
    EventType.LLM_CALL_ERROR: "llm.call.error",
    EventType.LLM_CALL_RETRIED: "llm.call.error",
    EventType.TOOL_CALL_STARTED: "tool.call.before",
    EventType.TOOL_CALL_COMPLETED: "tool.call.after",
    EventType.TOOL_CALL_ERROR: "tool.call.error",
    EventType.TOOL_CALL_RETRIED: "tool.call.error",
    EventType.PLANNING_STARTED: "agent.planning.before",
    EventType.PLANNING_COMPLETED: "agent.planning.after",
    EventType.CRITIQUE_STARTED: "agent.critique.before",
    EventType.CRITIQUE_COMPLETED: "agent.critique.after",
    EventType.SYNTHESIS_STARTED: "agent.synthesis.before",
    EventType.SYNTHESIS_COMPLETED: "agent.synthesis.after",
    EventType.CHECKPOINT_SAVED: "state.checkpoint.after",
    EventType.CHECKPOINT_RESTORED: "state.checkpoint.after",
    EventType.CUSTOM: "custom",
}


def _event_type_to_hook_name(event_type: EventType) -> str:
    """Convert an EventType to a dotted hook-style name for pattern matching."""
    return _EVENT_TYPE_TO_HOOK_NAME.get(event_type, event_type.value)
