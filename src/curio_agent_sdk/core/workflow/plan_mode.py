"""
Plan mode and todo management for agent workflows.

Supports plan-then-execute: enter plan mode (read-only exploration),
exit with a plan for approval, then approve to begin execution.
Todos track multi-step work and integrate with state/checkpoints.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from curio_agent_sdk.core.state import AgentState, StateExtension
from curio_agent_sdk.core.tools.tool import Tool

logger = logging.getLogger(__name__)

# Phase values for plan mode
PLAN_PHASE_PLANNING = "planning"
PLAN_PHASE_AWAITING_APPROVAL = "awaiting_approval"
PLAN_PHASE_EXECUTING = "executing"

TodoStatus = Literal["pending", "in_progress", "completed", "deleted"]


@dataclass
class PlanStep:
    """A single step in a plan."""
    title: str
    description: str = ""
    id: str = field(default_factory=lambda: f"step-{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "description": self.description, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            id=data.get("id", f"step-{uuid.uuid4().hex[:8]}"),
        )


@dataclass
class Plan:
    """A plan (list of steps) produced in plan mode."""
    steps: list[PlanStep] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(steps=steps, summary=data.get("summary", ""))


@dataclass
class PlanState(StateExtension):
    """
    State extension for plan mode: phase and current plan.

    Persisted in checkpoints via StateExtension protocol.
    """
    phase: str = PLAN_PHASE_EXECUTING  # planning | awaiting_approval | executing
    plan: Plan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "plan": self.plan.to_dict() if self.plan else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanState:
        plan_data = data.get("plan")
        plan = Plan.from_dict(plan_data) if plan_data else None
        return cls(phase=data.get("phase", PLAN_PHASE_EXECUTING), plan=plan)


@dataclass
class Todo(StateExtension):
    """
    A single todo item. Persisted in checkpoints via TodoState.
    """
    id: str
    subject: str
    description: str = ""
    status: TodoStatus = "pending"
    blocked_by: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "blocked_by": list(self.blocked_by),
            "blocks": list(self.blocks),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Todo:
        return cls(
            id=data.get("id", ""),
            subject=data.get("subject", ""),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            blocked_by=list(data.get("blocked_by", [])),
            blocks=list(data.get("blocks", [])),
        )


@dataclass
class TodoState(StateExtension):
    """
    State extension holding the list of todos for the current run.

    Persisted in checkpoints via StateExtension protocol.
    """
    todos: list[Todo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"todos": [t.to_dict() for t in self.todos]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoState:
        todos = [Todo.from_dict(t) for t in data.get("todos", [])]
        return cls(todos=todos)


class PlanMode:
    """
    Manages plan-then-execute workflows.

    - enter(): switch to plan mode (read-only exploration; only allowed tools are available).
    - exit(): save plan and switch to awaiting_approval (present plan to user).
    - approve(): user approved; switch to executing and restore full tools.

    Read-only tool restriction is applied by filtering state.tools / state.tool_schemas
    to only tools whose names are in read_only_tool_names. The full list is restored
    when exiting plan mode or when approving.
    """

    def __init__(
        self,
        read_only_tool_names: list[str] | None = None,
        tool_registry: Any = None,
    ):
        """
        Args:
            read_only_tool_names: Tool names allowed during plan mode. If None, no filtering (all tools).
            tool_registry: ToolRegistry to get full tool list for restore. Required for filtering.
        """
        self.read_only_tool_names = read_only_tool_names or []
        self.tool_registry = tool_registry
        self._state: AgentState | None = None

    def set_state(self, state: AgentState | None) -> None:
        """Set the current agent state (called by Runtime at run start)."""
        self._state = state

    def _get_plan_state(self, state: AgentState) -> PlanState:
        ps = state.get_ext(PlanState)
        if ps is None:
            ps = PlanState()
            state.set_ext(ps)
        return ps

    def is_in_plan_mode(self, state: AgentState | None = None) -> bool:
        """Return True if the agent is in planning phase (read-only)."""
        s = state or self._state
        if s is None:
            return False
        ps = s.get_ext(PlanState)
        return ps is not None and ps.phase == PLAN_PHASE_PLANNING

    def is_awaiting_approval(self, state: AgentState | None = None) -> bool:
        """Return True if the agent has a plan and is waiting for user approval."""
        s = state or self._state
        if s is None:
            return False
        ps = s.get_ext(PlanState)
        return ps is not None and ps.phase == PLAN_PHASE_AWAITING_APPROVAL

    async def enter(self, state: AgentState | None = None) -> None:
        """
        Switch agent to plan mode (read-only exploration).

        Sets phase to planning and optionally restricts state.tools to read_only_tool_names.
        """
        s = state or self._state
        if s is None:
            raise RuntimeError("PlanMode: no state set. Call set_state() from Runtime before using plan mode.")
        ps = self._get_plan_state(s)
        ps.phase = PLAN_PHASE_PLANNING
        ps.plan = None
        s.record_transition(PLAN_PHASE_PLANNING)
        if self.tool_registry and self.read_only_tool_names:
            allowed = {n for n in self.read_only_tool_names}
            s.tools = [t for t in self.tool_registry.tools if t.name in allowed]
            s.tool_schemas = [
                sch for t, sch in zip(self.tool_registry.tools, self.tool_registry.get_llm_schemas())
                if t.name in allowed
            ]
        logger.debug("Plan mode: entered (planning). Read-only tools: %s", self.read_only_tool_names)

    async def exit(self, state: AgentState | None = None, plan: Plan | None = None) -> None:
        """
        Exit plan mode and present the plan for approval.

        Saves the plan, sets phase to awaiting_approval, and restores full tools
        so the agent can report the plan (or wait for approve).
        """
        s = state or self._state
        if s is None:
            raise RuntimeError("PlanMode: no state set.")
        ps = self._get_plan_state(s)
        if plan is not None:
            ps.plan = plan
        ps.phase = PLAN_PHASE_AWAITING_APPROVAL
        s.record_transition(PLAN_PHASE_AWAITING_APPROVAL)
        if self.tool_registry:
            s.tools = self.tool_registry.tools
            s.tool_schemas = self.tool_registry.get_llm_schemas()
        logger.debug("Plan mode: exited. Awaiting approval. Plan steps: %d", len(ps.plan.steps) if ps.plan else 0)

    async def approve(self, state: AgentState | None = None) -> None:
        """
        User approved the plan; begin execution.

        Sets phase to executing and restores full tools.
        """
        s = state or self._state
        if s is None:
            raise RuntimeError("PlanMode: no state set.")
        ps = self._get_plan_state(s)
        ps.phase = PLAN_PHASE_EXECUTING
        s.record_transition(PLAN_PHASE_EXECUTING)
        if self.tool_registry:
            s.tools = self.tool_registry.tools
            s.tool_schemas = self.tool_registry.get_llm_schemas()
        logger.debug("Plan mode: approved. Executing.")

    def get_plan(self, state: AgentState | None = None) -> Plan | None:
        """Return the current plan if any."""
        s = state or self._state
        if s is None:
            return None
        ps = s.get_ext(PlanState)
        return ps.plan if ps else None


class TodoManager:
    """
    Manages todo items during agent execution.

    Todos are stored in state via TodoState extension and persisted in checkpoints.
    """

    def __init__(self):
        self._state: AgentState | None = None

    def set_state(self, state: AgentState | None) -> None:
        """Set the current agent state (called by Runtime at run start)."""
        self._state = state

    def _get_todo_state(self, state: AgentState) -> TodoState:
        ts = state.get_ext(TodoState)
        if ts is None:
            ts = TodoState()
            state.set_ext(ts)
        return ts

    async def create(self, subject: str, description: str = "") -> Todo:
        """Create a new todo. Returns the created Todo."""
        s = self._state
        if s is None:
            raise RuntimeError("TodoManager: no state set. Call set_state() from Runtime before using todos.")
        ts = self._get_todo_state(s)
        todo_id = f"todo-{uuid.uuid4().hex[:8]}"
        todo = Todo(id=todo_id, subject=subject, description=description, status="pending")
        ts.todos.append(todo)
        return todo

    async def update(self, todo_id: str, status: TodoStatus) -> None:
        """Update a todo's status."""
        s = self._state
        if s is None:
            raise RuntimeError("TodoManager: no state set.")
        ts = self._get_todo_state(s)
        for t in ts.todos:
            if t.id == todo_id:
                t.status = status
                return
        raise ValueError(f"Todo not found: {todo_id}")

    async def list(self) -> list[Todo]:
        """List all todos (excluding deleted by default)."""
        s = self._state
        if s is None:
            return []
        ts = self._get_todo_state(s)
        return [t for t in ts.todos if t.status != "deleted"]

    async def list_all(self) -> list[Todo]:
        """List all todos including deleted."""
        s = self._state
        if s is None:
            return []
        ts = self._get_todo_state(s)
        return list(ts.todos)

    async def get(self, todo_id: str) -> Todo | None:
        """Get a todo by id."""
        s = self._state
        if s is None:
            return None
        ts = self._get_todo_state(s)
        for t in ts.todos:
            if t.id == todo_id:
                return t
        return None


def get_plan_mode_tools(
    plan_mode: PlanMode,
    todo_manager: TodoManager,
) -> list[Tool]:
    """
    Build tools for plan mode and todos: enter_plan_mode, exit_plan_mode, approve_plan,
    create_todo, update_todo, list_todos, get_todo.
    """
    from curio_agent_sdk.core.tools.tool import tool as tool_decorator

    @tool_decorator
    async def enter_plan_mode() -> str:
        """Switch to plan mode for designing an implementation approach. In plan mode only read-only tools (e.g. read, search) are available. Use this before making changes to explore and design a plan, then call exit_plan_mode to submit the plan for approval."""
        await plan_mode.enter()
        return "Entered plan mode. Only read-only tools are available. Design your approach, then call exit_plan_mode with your plan."

    @tool_decorator
    async def exit_plan_mode(
        plan_summary: str = "",
        steps_json: str = "",
    ) -> str:
        """Exit plan mode and submit the plan for user approval. Provide plan_summary (optional) and steps_json: a JSON array of objects with 'title' and 'description' for each step, e.g. [{\"title\": \"Step 1\", \"description\": \"...\"}]."""
        import json
        steps = []
        if steps_json.strip():
            try:
                raw = json.loads(steps_json)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            steps.append(PlanStep(
                                title=item.get("title", ""),
                                description=item.get("description", ""),
                            ))
            except json.JSONDecodeError:
                pass
        plan = Plan(steps=steps, summary=plan_summary or "")
        await plan_mode.exit(plan=plan)
        return f"Plan submitted for approval ({len(plan.steps)} steps). Summary: {plan.summary or 'N/A'}. User must call approve_plan to begin execution."

    @tool_decorator
    async def approve_plan() -> str:
        """Approve the current plan and begin execution. Only the user or orchestrator should call this; it restores full tool access."""
        await plan_mode.approve()
        return "Plan approved. Full tools restored. Proceeding with execution."

    @tool_decorator
    async def create_todo(subject: str, description: str = "") -> str:
        """Create a todo item to track progress. Use for multi-step tasks. Returns the todo id."""
        todo = await todo_manager.create(subject, description)
        return f"Created todo: {todo.id} — {todo.subject}"

    @tool_decorator
    async def update_todo(todo_id: str, status: str = "in_progress") -> str:
        """Update a todo's status. Status must be one of: pending, in_progress, completed, deleted."""
        if status not in ("pending", "in_progress", "completed", "deleted"):
            return f"Invalid status: {status}. Use pending, in_progress, completed, or deleted."
        await todo_manager.update(todo_id, status)
        return f"Updated todo {todo_id} to {status}."

    @tool_decorator
    async def list_todos() -> str:
        """List all active todos (pending, in_progress, completed). Returns a summary string."""
        todos = await todo_manager.list()
        if not todos:
            return "No todos."
        lines = [f"- [{t.id}] {t.subject} ({t.status})" for t in todos]
        return "\n".join(lines)

    @tool_decorator
    async def get_todo(todo_id: str) -> str:
        """Get a single todo by id."""
        todo = await todo_manager.get(todo_id)
        if todo is None:
            return f"Todo not found: {todo_id}"
        return f"[{todo.id}] {todo.subject} — {todo.status}\n{todo.description or ''}"

    return [
        enter_plan_mode,
        exit_plan_mode,
        approve_plan,
        create_todo,
        update_todo,
        list_todos,
        get_todo,
    ]
