"""
Event types and stream events for agent observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from datetime import datetime


class EventType(str, Enum):
    """Types of events emitted during agent execution."""
    # Run lifecycle
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_ERROR = "run_error"
    RUN_CANCELLED = "run_cancelled"
    RUN_TIMEOUT = "run_timeout"

    # Loop iterations
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"

    # LLM calls
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    LLM_CALL_ERROR = "llm_call_error"
    LLM_CALL_RETRIED = "llm_call_retried"

    # Tool execution
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_ERROR = "tool_call_error"
    TOOL_CALL_RETRIED = "tool_call_retried"

    # Agent-specific phases
    PLANNING_STARTED = "planning_started"
    PLANNING_COMPLETED = "planning_completed"
    CRITIQUE_STARTED = "critique_started"
    CRITIQUE_COMPLETED = "critique_completed"
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETED = "synthesis_completed"

    # Memory & state
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    # Custom
    CUSTOM = "custom"


@dataclass
class AgentEvent:
    """
    An event emitted during agent execution.

    Used for observability, logging, and middleware.
    """
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    agent_id: str = ""
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "iteration": self.iteration,
        }


@dataclass
class StreamEvent:
    """
    Events emitted during streaming agent execution.

    These are the user-facing events for real-time observation.
    """
    type: Literal[
        "text_delta",
        "tool_call_start",
        "tool_call_end",
        "thinking",
        "iteration_start",
        "iteration_end",
        "error",
        "done",
    ]
    data: Any = None
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    error: str | None = None
    iteration: int = 0
