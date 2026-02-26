"""
Agent state - the mutable context passed through the agent loop.

Supports typed state extensions (e.g. PlanState for plan mode) and
state transition tracking for custom loops and orchestrators.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from curio_agent_sdk.models.llm import Message, ToolSchema
from curio_agent_sdk.core.tools.tool import Tool

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class StateExtension(Protocol):
    """
    Protocol for state extensions that support checkpoint serialization.

    Implement to_dict() and from_dict() so extensions are saved/restored
    when state is checkpointed. Extensions are keyed by type module + qualname.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize this extension to a JSON-serializable dict."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Deserialize an instance from a dict (from to_dict())."""
        ...


@dataclass
class AgentState:
    """
    Mutable state container passed through the agent loop.

    This holds everything the loop needs to operate:
    - The conversation message history
    - Available tools
    - Iteration counter
    - Cancellation signal
    - Metrics accumulators
    """
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    tool_schemas: list[ToolSchema] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 25
    metadata: dict[str, Any] = field(default_factory=dict)

    # Metrics
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Timing
    start_time: float = field(default_factory=time.monotonic)

    # Control
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _done: bool = False
    _last_finish_reason: str = ""

    # Typed state extensions (e.g. PlanState for plan mode)
    _extensions: dict[type, Any] = field(default_factory=dict, repr=False)

    # State transition tracking (phase name -> monotonic timestamp)
    _transition_history: list[tuple[str, float]] = field(default_factory=list, repr=False)
    current_phase: str = ""

    def get_ext(self, ext_type: type[T]) -> T | None:
        """
        Get a typed state extension by type.

        Example:
            plan_state = state.get_ext(PlanState)
        """
        return self._extensions.get(ext_type)

    def set_ext(self, ext: Any) -> None:
        """
        Attach a state extension. Stored by type(ext).
        If the extension implements StateExtension (to_dict/from_dict),
        it will be included in checkpoints.
        """
        self._extensions[type(ext)] = ext

    def record_transition(self, phase: str) -> None:
        """Record a state transition (e.g. planning -> executing)."""
        now = time.monotonic()
        self._transition_history.append((phase, now))
        self.current_phase = phase

    def get_transition_history(self) -> list[tuple[str, float]]:
        """Return the list of (phase_name, monotonic_timestamp) transitions."""
        return list(self._transition_history)

    def set_transition_history(self, history: list[tuple[str, float]]) -> None:
        """Restore transition history (e.g. after loading from checkpoint)."""
        self._transition_history = list(history)
        self.current_phase = history[-1][0] if history else ""

    def get_extensions_for_checkpoint(self) -> dict[str, dict[str, Any]]:
        """
        Serialize extensions that implement StateExtension for checkpointing.
        Keys are f"{type.__module__}.{type.__qualname__}"; values include
        __module__ and __qualname__ for deserialization.
        """
        out: dict[str, dict[str, Any]] = {}
        for typ, value in self._extensions.items():
            if not isinstance(value, StateExtension):
                continue
            try:
                data = dict(value.to_dict())
                data["__module__"] = typ.__module__
                data["__qualname__"] = typ.__qualname__
                key = f"{typ.__module__}.{typ.__qualname__}"
                out[key] = data
            except Exception as e:
                logger.warning("Failed to serialize state extension %s: %s", typ, e)
        return out

    def set_extensions_from_checkpoint(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Restore extensions from checkpoint data.
        Resolves types by __module__ and __qualname__; skips unknown types.
        """
        for key, value in data.items():
            try:
                module_name = value.get("__module__")
                qualname = value.get("__qualname__")
                if not module_name or not qualname:
                    continue
                module = importlib.import_module(module_name)
                typ: Any = module
                for part in qualname.split("."):
                    typ = getattr(typ, part)
                payload = {k: v for k, v in value.items() if k not in ("__module__", "__qualname__")}
                instance = typ.from_dict(payload)
                self.set_ext(instance)
            except Exception as e:
                logger.warning("Failed to restore state extension %s: %s", key, e)

    def cancel(self):
        """Signal the agent to stop after the current step."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def is_done(self) -> bool:
        return self._done

    def mark_done(self):
        """Mark the loop as complete."""
        self._done = True

    def add_message(self, message: Message):
        """Append a message to the conversation history."""
        self.messages.append(message)

    def add_messages(self, messages: list[Message]):
        """Append multiple messages."""
        self.messages.extend(messages)

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record an LLM call in the metrics."""
        self.total_llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_tool_calls(self, count: int = 1):
        """Record tool call(s) in the metrics."""
        self.total_tool_calls += count

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds since the state was created."""
        return time.monotonic() - self.start_time

    @property
    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def assistant_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]
