"""
Unit tests for curio_agent_sdk.core.workflow.plan_mode — PlanStep, Plan, PlanState,
PlanMode, Todo, TodoState, TodoManager, get_plan_mode_tools.
"""

import pytest

from curio_agent_sdk.core.state.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.core.tools.registry import ToolRegistry
from curio_agent_sdk.core.workflow.plan_mode import (
    PlanStep,
    Plan,
    PlanState,
    Todo,
    TodoState,
    PlanMode,
    TodoManager,
    get_plan_mode_tools,
    PLAN_PHASE_PLANNING,
    PLAN_PHASE_AWAITING_APPROVAL,
    PLAN_PHASE_EXECUTING,
)


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanStepCreation:
    def test_plan_step_creation(self):
        """PlanStep dataclass: title, description, id."""
        step = PlanStep(title="Step 1", description="Do something")
        assert step.title == "Step 1"
        assert step.description == "Do something"
        assert step.id.startswith("step-")
        assert len(step.id) == len("step-") + 8

    def test_plan_step_to_dict_from_dict(self):
        """PlanStep serialization round-trip."""
        step = PlanStep(title="T", description="D", id="step-abc12345")
        d = step.to_dict()
        assert d == {"title": "T", "description": "D", "id": "step-abc12345"}
        back = PlanStep.from_dict(d)
        assert back.title == step.title and back.description == step.description and back.id == step.id


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanCreation:
    def test_plan_creation(self):
        """Plan with steps and summary."""
        steps = [
            PlanStep(title="A", description="First", id="s1"),
            PlanStep(title="B", description="Second", id="s2"),
        ]
        plan = Plan(steps=steps, summary="Two steps")
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "A"
        assert plan.summary == "Two steps"

    def test_plan_to_dict_from_dict(self):
        """Plan serialization round-trip."""
        plan = Plan(
            steps=[PlanStep(title="X", id="x1")],
            summary="Summary",
        )
        d = plan.to_dict()
        assert "steps" in d and "summary" in d
        back = Plan.from_dict(d)
        assert len(back.steps) == 1 and back.steps[0].title == "X" and back.summary == "Summary"


# ---------------------------------------------------------------------------
# PlanState
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanStateCreation:
    def test_plan_state_creation(self):
        """PlanState extension: phase and plan."""
        ps = PlanState(phase=PLAN_PHASE_PLANNING, plan=None)
        assert ps.phase == PLAN_PHASE_PLANNING
        assert ps.plan is None
        plan = Plan(steps=[PlanStep(title="P", id="p1")], summary="S")
        ps2 = PlanState(phase=PLAN_PHASE_AWAITING_APPROVAL, plan=plan)
        assert ps2.plan is not None and len(ps2.plan.steps) == 1
        d = ps2.to_dict()
        assert d["phase"] == PLAN_PHASE_AWAITING_APPROVAL
        back = PlanState.from_dict(d)
        assert back.phase == ps2.phase and back.plan is not None


# ---------------------------------------------------------------------------
# PlanMode — enter / exit / approve
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanModePhases:
    @pytest.fixture
    def state(self):
        return AgentState()

    @pytest.fixture
    def plan_mode(self):
        return PlanMode(read_only_tool_names=None, tool_registry=None)

    @pytest.mark.asyncio
    async def test_plan_mode_enter_planning(self, state, plan_mode):
        """Enter planning phase."""
        plan_mode.set_state(state)
        await plan_mode.enter(state)
        ps = state.get_ext(PlanState)
        assert ps is not None
        assert ps.phase == PLAN_PHASE_PLANNING
        assert ps.plan is None
        assert state.current_phase == PLAN_PHASE_PLANNING

    @pytest.mark.asyncio
    async def test_plan_mode_exit_returns_plan(self, state, plan_mode):
        """Exit saves plan and sets awaiting_approval; get_plan returns it."""
        plan_mode.set_state(state)
        await plan_mode.enter(state)
        steps = [PlanStep(title="Run tests", id="s1")]
        plan = Plan(steps=steps, summary="Run the test suite")
        await plan_mode.exit(state, plan=plan)
        ps = state.get_ext(PlanState)
        assert ps.phase == PLAN_PHASE_AWAITING_APPROVAL
        assert ps.plan is not None and len(ps.plan.steps) == 1
        got = plan_mode.get_plan(state)
        assert got is not None and got.summary == "Run the test suite"

    @pytest.mark.asyncio
    async def test_plan_mode_execute_plan(self, state, plan_mode):
        """Approve sets phase to executing."""
        plan_mode.set_state(state)
        await plan_mode.enter(state)
        await plan_mode.exit(state, plan=Plan(steps=[], summary="Go"))
        await plan_mode.approve(state)
        ps = state.get_ext(PlanState)
        assert ps.phase == PLAN_PHASE_EXECUTING
        assert state.current_phase == PLAN_PHASE_EXECUTING

    @pytest.mark.asyncio
    async def test_plan_mode_read_only_tools(self, state):
        """During planning, state.tools restricted to read_only_tool_names."""
        async def read_tool():
            return "read"
        async def write_tool():
            return "write"
        t_read = Tool(func=read_tool, name="read_only_tool")
        t_write = Tool(func=write_tool, name="write_tool")
        registry = ToolRegistry([t_read, t_write])
        plan_mode = PlanMode(read_only_tool_names=["read_only_tool"], tool_registry=registry)
        plan_mode.set_state(state)
        state.tools = [t_read, t_write]
        state.tool_schemas = registry.get_llm_schemas()
        await plan_mode.enter(state)
        assert len(state.tools) == 1
        assert state.tools[0].name == "read_only_tool"
        assert len(state.tool_schemas) == 1
        await plan_mode.exit(state)
        assert len(state.tools) == 2
        assert set(t.name for t in state.tools) == {"read_only_tool", "write_tool"}

    @pytest.mark.asyncio
    async def test_plan_mode_enter_no_state_raises(self, plan_mode):
        """enter() with no state set raises RuntimeError."""
        with pytest.raises(RuntimeError, match="no state set"):
            await plan_mode.enter()

    def test_is_in_plan_mode(self, state, plan_mode):
        """is_in_plan_mode True only when phase is planning."""
        plan_mode.set_state(state)
        assert plan_mode.is_in_plan_mode(state) is False
        state.set_ext(PlanState(phase=PLAN_PHASE_PLANNING, plan=None))
        assert plan_mode.is_in_plan_mode(state) is True
        state.get_ext(PlanState).phase = PLAN_PHASE_EXECUTING
        assert plan_mode.is_in_plan_mode(state) is False

    def test_is_awaiting_approval(self, state, plan_mode):
        """is_awaiting_approval True only when phase is awaiting_approval."""
        plan_mode.set_state(state)
        assert plan_mode.is_awaiting_approval(state) is False
        state.set_ext(PlanState(phase=PLAN_PHASE_AWAITING_APPROVAL, plan=Plan(steps=[], summary="")))
        assert plan_mode.is_awaiting_approval(state) is True


# ---------------------------------------------------------------------------
# TodoManager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTodoManager:
    @pytest.fixture
    def state(self):
        return AgentState()

    @pytest.fixture
    def todo_manager(self):
        return TodoManager()

    @pytest.mark.asyncio
    async def test_todo_manager_create(self, state, todo_manager):
        """Create todo item."""
        todo_manager.set_state(state)
        todo = await todo_manager.create("Implement feature", "Add tests")
        assert todo.subject == "Implement feature"
        assert todo.description == "Add tests"
        assert todo.status == "pending"
        assert todo.id.startswith("todo-")

    @pytest.mark.asyncio
    async def test_todo_manager_get(self, state, todo_manager):
        """Get todo by ID."""
        todo_manager.set_state(state)
        created = await todo_manager.create("Task", "")
        got = await todo_manager.get(created.id)
        assert got is not None and got.id == created.id and got.subject == "Task"
        assert await todo_manager.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_todo_manager_update_status(self, state, todo_manager):
        """Update todo status."""
        todo_manager.set_state(state)
        todo = await todo_manager.create("Task", "")
        await todo_manager.update(todo.id, "in_progress")
        t = await todo_manager.get(todo.id)
        assert t.status == "in_progress"
        await todo_manager.update(todo.id, "completed")
        t = await todo_manager.get(todo.id)
        assert t.status == "completed"

    @pytest.mark.asyncio
    async def test_todo_manager_list(self, state, todo_manager):
        """List todos (excludes deleted)."""
        todo_manager.set_state(state)
        await todo_manager.create("A", "")
        await todo_manager.create("B", "")
        todos = await todo_manager.list()
        assert len(todos) == 2
        await todo_manager.update(todos[0].id, "deleted")
        listed = await todo_manager.list()
        assert len(listed) == 1
        assert listed[0].subject == "B"

    @pytest.mark.asyncio
    async def test_todo_manager_list_by_status(self, state, todo_manager):
        """Filter by status via list vs list_all."""
        todo_manager.set_state(state)
        t1 = await todo_manager.create("P", "")
        t2 = await todo_manager.create("Q", "")
        await todo_manager.update(t1.id, "completed")
        listed = await todo_manager.list()
        assert len(listed) == 2  # list() returns non-deleted
        all_list = await todo_manager.list_all()
        assert len(all_list) == 2
        completed = [t for t in listed if t.status == "completed"]
        assert len(completed) == 1 and completed[0].subject == "P"

    @pytest.mark.asyncio
    async def test_todo_status_transitions(self, state, todo_manager):
        """pending → in_progress → completed."""
        todo_manager.set_state(state)
        todo = await todo_manager.create("Work", "")
        assert todo.status == "pending"
        await todo_manager.update(todo.id, "in_progress")
        todo = await todo_manager.get(todo.id)
        assert todo.status == "in_progress"
        await todo_manager.update(todo.id, "completed")
        todo = await todo_manager.get(todo.id)
        assert todo.status == "completed"

    @pytest.mark.asyncio
    async def test_todo_manager_update_not_found_raises(self, state, todo_manager):
        """update with unknown id raises ValueError."""
        todo_manager.set_state(state)
        with pytest.raises(ValueError, match="not found"):
            await todo_manager.update("todo-nonexistent", "completed")

    @pytest.mark.asyncio
    async def test_todo_manager_create_no_state_raises(self, todo_manager):
        """create with no state raises RuntimeError."""
        with pytest.raises(RuntimeError, match="no state set"):
            await todo_manager.create("X", "")


# ---------------------------------------------------------------------------
# get_plan_mode_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanModeTools:
    def test_get_plan_mode_tools_returns_list(self):
        """get_plan_mode_tools returns list of Tool callables."""
        plan_mode = PlanMode()
        todo_mgr = TodoManager()
        tools = get_plan_mode_tools(plan_mode, todo_mgr)
        names = [t.name for t in tools]
        assert "enter_plan_mode" in names
        assert "exit_plan_mode" in names
        assert "approve_plan" in names
        assert "create_todo" in names
        assert "update_todo" in names
        assert "list_todos" in names
        assert "get_todo" in names
        assert len(tools) == 7
