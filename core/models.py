"""
Core data models for the Curio Agent SDK.

These models are used for tracking agent runs, events, and LLM usage
for observability and debugging purposes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
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


class EventType(str, Enum):
    """Types of events that can occur during an agent run."""
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_ERROR = "run_error"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    PLANNING_STARTED = "planning_started"
    PLANNING_COMPLETED = "planning_completed"
    ACTION_EXECUTION_STARTED = "action_execution_started"
    ACTION_EXECUTION_COMPLETED = "action_execution_completed"
    CRITIQUE_STARTED = "critique_started"
    CRITIQUE_COMPLETED = "critique_completed"
    CRITIQUE_RESULT = "critique_result"
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    SYNTHESIS_RESULT = "synthesis_result"
    OBJECT_STORED = "object_stored"
    OBJECT_NOT_FOUND = "object_not_found"
    TOOL_REGISTERED = "tool_registered"
    SUBAGENT_RUN_STARTED = "subagent_run_started"
    SUBAGENT_RUN_COMPLETED = "subagent_run_completed"
    CUSTOM = "custom"


@dataclass
class AgentRun:
    """
    Represents a single agent run with all its metadata.

    An agent run is a complete execution cycle of an agent, starting from
    receiving an objective and ending with a synthesis of results.

    Attributes:
        agent_id: Unique identifier for the agent instance
        run_id: Unique identifier for this specific run
        agent_name: Human-readable name of the agent
        objective: The objective/goal for this run
        additional_context: Any additional context provided for the run
        started_at: Timestamp when the run started
        finished_at: Timestamp when the run finished (None if still running)
        total_iterations: Number of plan-execute-critique iterations
        final_synthesis_output: The final synthesis summary
        execution_history: JSON string of the full execution history
        status: Current status of the run
        error_message: Error message if status is ERROR
        metadata: Additional metadata as JSON
    """
    agent_id: str
    run_id: str
    agent_name: str = ""
    objective: str = ""
    additional_context: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    total_iterations: int = 0
    final_synthesis_output: Optional[str] = None
    execution_history: Optional[str] = None
    status: str = AgentRunStatus.PENDING.value
    error_message: Optional[str] = None
    metadata: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "objective": self.objective,
            "additional_context": self.additional_context,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_iterations": self.total_iterations,
            "final_synthesis_output": self.final_synthesis_output,
            "execution_history": self.execution_history,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRun":
        """Create from dictionary representation."""
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
            final_synthesis_output=data.get("final_synthesis_output"),
            execution_history=data.get("execution_history"),
            status=data.get("status", AgentRunStatus.PENDING.value),
            error_message=data.get("error_message"),
            metadata=data.get("metadata"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class AgentRunEvent:
    """
    Represents a single event during an agent run.

    Events are logged throughout the agent's execution to provide
    detailed observability into what the agent is doing.

    Attributes:
        agent_id: Unique identifier for the agent instance
        run_id: Unique identifier for the run this event belongs to
        agent_name: Human-readable name of the agent
        timestamp: When the event occurred
        event_type: Type of event (from EventType enum)
        data: JSON string of event-specific data
    """
    agent_id: str
    run_id: str
    agent_name: str = ""
    timestamp: Optional[datetime] = None
    event_type: str = ""
    data: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "data": self.data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRunEvent":
        """Create from dictionary representation."""
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

    def get_data_dict(self) -> Dict[str, Any]:
        """Parse and return the data field as a dictionary."""
        if self.data:
            try:
                return json.loads(self.data)
            except json.JSONDecodeError:
                return {"raw": self.data}
        return {}


@dataclass
class AgentLLMUsage:
    """
    Tracks LLM usage for cost monitoring and debugging.

    Every LLM call made by an agent is recorded with this model,
    allowing for detailed analysis of token usage, costs, and performance.

    Attributes:
        agent_id: Unique identifier for the agent instance
        run_id: Unique identifier for the run this call belongs to
        provider: LLM provider name (e.g., "openai", "anthropic")
        model: Model name used for the call
        prompt: The input prompt sent to the LLM
        prompt_length: Character length of the prompt
        input_params: JSON string of input parameters (temperature, max_tokens, etc.)
        input_tokens: Number of input tokens (if available)
        output_tokens: Number of output tokens (if available)
        response_content: The response from the LLM
        response_length: Character length of the response
        usage_metrics: JSON string of additional usage metrics
        status: "success" or "error"
        error_message: Error message if status is "error"
        latency_ms: Response latency in milliseconds
    """
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    provider: str = ""
    model: str = ""
    prompt: str = ""
    prompt_length: int = 0
    input_params: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_content: Optional[str] = None
    response_length: Optional[int] = None
    usage_metrics: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "prompt_length": self.prompt_length,
            "input_params": self.input_params,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "response_content": self.response_content,
            "response_length": self.response_length,
            "usage_metrics": self.usage_metrics,
            "status": self.status,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentLLMUsage":
        """Create from dictionary representation."""
        return cls(
            id=data.get("id"),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            prompt=data.get("prompt", ""),
            prompt_length=data.get("prompt_length", 0),
            input_params=data.get("input_params"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            response_content=data.get("response_content"),
            response_length=data.get("response_length"),
            usage_metrics=data.get("usage_metrics"),
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
            latency_ms=data.get("latency_ms"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )

    def get_total_tokens(self) -> int:
        """Get total tokens used (input + output)."""
        input_t = self.input_tokens or 0
        output_t = self.output_tokens or 0
        return input_t + output_t

    def get_input_params_dict(self) -> Dict[str, Any]:
        """Parse and return input_params as a dictionary."""
        if self.input_params:
            try:
                return json.loads(self.input_params)
            except json.JSONDecodeError:
                return {}
        return {}


@dataclass
class CritiqueResult:
    """
    Result of a critique phase in the agentic loop.

    Attributes:
        status: "done" to stop iteration, "continue" to keep going
        critique_summary: Summary of the critique evaluation
        recommendations: Recommendations for next steps
    """
    status: str  # "done" or "continue"
    critique_summary: str = ""
    recommendations: str = ""

    def should_continue(self) -> bool:
        """Check if the agent should continue iterating."""
        return self.status.lower() == "continue"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "critique_summary": self.critique_summary,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CritiqueResult":
        """Create from dictionary representation."""
        return cls(
            status=data.get("status", "done"),
            critique_summary=data.get("critique_summary", ""),
            recommendations=data.get("recommendations", ""),
        )


@dataclass
class PlannedAction:
    """
    A single action planned by the agent.

    Attributes:
        action: Name of the action/tool to execute
        args: Arguments to pass to the action
    """
    action: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action": self.action,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedAction":
        """Create from dictionary representation."""
        return cls(
            action=data.get("action", ""),
            args=data.get("args", {}),
        )


@dataclass
class PlanResult:
    """
    Result of a planning phase in the agentic loop.

    Attributes:
        plan: List of planned actions to execute
        notes: Notes about the planning process
        debug_info: Debug information for improvement suggestions
    """
    plan: List[PlannedAction] = field(default_factory=list)
    notes: str = ""
    debug_info: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "plan": [a.to_dict() for a in self.plan],
            "notes": self.notes,
            "debugInfo": self.debug_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanResult":
        """Create from dictionary representation."""
        plan_data = data.get("plan", [])
        return cls(
            plan=[PlannedAction.from_dict(a) for a in plan_data],
            notes=data.get("notes", ""),
            debug_info=data.get("debugInfo", data.get("debug_info", "")),
        )


@dataclass
class SynthesisResult:
    """
    Result of the synthesis phase at the end of an agent run.

    Attributes:
        synthesis_summary: Summary of what was accomplished
    """
    synthesis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "synthesis_summary": self.synthesis_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesisResult":
        """Create from dictionary representation."""
        return cls(
            synthesis_summary=data.get("synthesis_summary", ""),
        )


@dataclass
class AgentRunResult:
    """
    Complete result of an agent run.

    Attributes:
        status: Final status ("done", "error", etc.)
        synthesis_summary: Summary of what was accomplished
        total_iterations: Number of iterations completed
        run_id: Unique identifier for this run
        error: Error message if status is "error"
        execution_history: Full execution history
    """
    status: str
    synthesis_summary: str = ""
    total_iterations: int = 0
    run_id: str = ""
    error: Optional[str] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

    def is_success(self) -> bool:
        """Check if the run completed successfully."""
        return self.status == "done"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "synthesis_summary": self.synthesis_summary,
            "total_iterations": self.total_iterations,
            "run_id": self.run_id,
            "error": self.error,
            "execution_history": self.execution_history,
        }
