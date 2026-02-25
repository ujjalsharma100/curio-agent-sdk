"""
Agent data models for runs, events, results, and state tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from enum import Enum
import json


class AgentRunStatus(str, Enum):
    """Status of an agent run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AgentRun:
    """
    Represents a single agent run record for persistence/observability.
    """
    agent_id: str
    run_id: str
    agent_name: str = ""
    objective: str = ""
    additional_context: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_iterations: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    final_output: str | None = None
    execution_history: str | None = None  # JSON
    status: str = AgentRunStatus.PENDING.value
    error_message: str | None = None
    metadata: str | None = None  # JSON
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "objective": self.objective,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_iterations": self.total_iterations,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "final_output": self.final_output,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRun:
        return cls(
            id=data.get("id"),
            agent_id=data["agent_id"],
            run_id=data["run_id"],
            agent_name=data.get("agent_name", ""),
            objective=data.get("objective", ""),
            additional_context=data.get("additional_context"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            finished_at=datetime.fromisoformat(data["finished_at"]) if data.get("finished_at") else None,
            total_iterations=data.get("total_iterations", 0),
            total_llm_calls=data.get("total_llm_calls", 0),
            total_tool_calls=data.get("total_tool_calls", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            final_output=data.get("final_output"),
            execution_history=data.get("execution_history"),
            status=data.get("status", AgentRunStatus.PENDING.value),
            error_message=data.get("error_message"),
            metadata=data.get("metadata"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class AgentRunEvent:
    """A single event during an agent run, for persistence."""
    agent_id: str
    run_id: str
    agent_name: str = ""
    timestamp: datetime | None = None
    event_type: str = ""
    data: str | None = None  # JSON
    id: int | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRunEvent:
        return cls(
            id=data.get("id"),
            agent_id=data["agent_id"],
            run_id=data["run_id"],
            agent_name=data.get("agent_name", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            event_type=data.get("event_type", ""),
            data=data.get("data"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )

    def get_data_dict(self) -> dict[str, Any]:
        if self.data:
            try:
                return json.loads(self.data)
            except json.JSONDecodeError:
                return {"raw": self.data}
        return {}


@dataclass
class AgentLLMUsage:
    """Tracks a single LLM call for cost/performance monitoring."""
    agent_id: str | None = None
    run_id: str | None = None
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    status: str = "success"
    error_message: str | None = None
    metadata: str | None = None  # JSON
    id: int | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error_message": self.error_message,
        }

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentRunResult:
    """Complete result of an agent run, returned to the caller."""
    status: str  # "completed", "error", "cancelled", "timeout"
    output: str = ""
    total_iterations: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    run_id: str = ""
    error: str | None = None
    messages: list = field(default_factory=list)  # Full message history

    @property
    def is_success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "total_iterations": self.total_iterations,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "run_id": self.run_id,
            "error": self.error,
        }
